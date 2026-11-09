"""
SEW-ResNet18 for Facial Emotion Recognition
Spiking Neural Network implementation using SpikingJelly

References:
    Fang et al. (2021a) "Deep Residual Learning in Spiking Neural Networks"
        NeurIPS 2021. https://arxiv.org/abs/2102.04159          (SEW-ResNet)
    Fang et al. (2021b) "Incorporating Learnable Membrane Time Constant to
        Enhance Learning of Spiking Neural Networks", ICCV 2021  (ATan surrogate)
    Fang et al. (2023) "SpikingJelly", Science Advances 9(40), eadi1480.

Changes with respect to the first version
-----------------------------------------
* No silent fallback: if SpikingJelly cannot build the network, an error is
  raised (a silent fallback to a plain ResNet18 would invalidate any SNN result).
* ``activation`` argument: 'lif' (spiking, default) or 'relu'. The 'relu' mode
  builds the very same SEW-ResNet18 topology with ReLU instead of LIF neurons
  (T forced to 1) and serves as a control to isolate the effect of spiking.
* ``normalize_input`` argument: per-sample min-max rescaling to [0, 1] can be
  switched off, so that ANN and SNN receive identical inputs.
* Neuron / surrogate hyper-parameters are explicit constructor arguments
  (defaults equal the values previously used implicitly: tau=2, V_th=1,
  V_reset=0, ATan alpha=2, detach_reset=True).
* ``backend`` argument ('torch' or 'cupy') for the LIF neurons.
* ``set_num_timesteps`` to run T-ablations without rebuilding the model.
* ``SpikeStatsRecorder``: context manager that records, per spiking layer,
  the number of spikes and of neurons, without keeping activations in memory
  and without leaking hooks.
* Neurons are reset at the start AND at the end of every forward pass.

Parameter names are unchanged (``backbone.*``): checkpoints saved with the
previous version still load with ``load_state_dict``.
"""

import copy
from collections import OrderedDict

import torch
import torch.nn as nn

import spikingjelly
from spikingjelly.activation_based import layer, neuron, surrogate, functional
from spikingjelly.activation_based.model import sew_resnet

__all__ = [
    "SEWResNet18EmotionRecognizer",
    "create_sew_resnet18",
    "reset_snn_state",
    "test_sew_resnet",
    "attach_spike_hooks",
    "SpikeStatsRecorder",
    "get_spiking_layers",
    "verify_snn_compliance",
]


def _spikingjelly_version():
    """Installed SpikingJelly version (package metadata, then module attribute)."""
    try:
        from importlib.metadata import version
        return version("spikingjelly")
    except Exception:
        return getattr(spikingjelly, '__version__', 'unknown')


def _relu_factory(**kwargs):
    """Neuron factory used by the 'relu' control: ignores spiking kwargs."""
    return nn.ReLU()


class SEWResNet18EmotionRecognizer(nn.Module):
    """
    SEW-ResNet18 for facial emotion recognition on static grayscale images.

    Implements the Spike-Element-Wise ResNet18 architecture (Fang et al., 2021a)
    with direct encoding for static image datasets, ATan surrogate gradient
    (Fang et al., 2021b), and multi-step temporal processing.

    Args:
        num_classes     (int)  : Number of emotion classes. Default: 7.
        num_timesteps   (int)  : Number of simulation timesteps T. Default: 8.
        encoding_type   (str)  : Encoding label for logging. Default: 'direct'.
        activation      (str)  : 'lif' (spiking network) or 'relu' (control with
                                 the same topology, T forced to 1). Default: 'lif'.
        normalize_input (bool) : Apply per-sample min-max rescaling to [0, 1]
                                 before encoding. Default: True (original behaviour).
        tau, v_threshold, v_reset : LIF parameters. Defaults: 2.0, 1.0, 0.0.
        alpha           (float): ATan surrogate sharpness. Default: 2.0.
        detach_reset    (bool) : Detach the reset path from the graph. Default: True.
        backend         (str)  : 'torch' or 'cupy' (LIF only). Default: 'torch'.
    """

    def __init__(self, num_classes=7, num_timesteps=8, encoding_type="direct",
                 activation="lif", normalize_input=True,
                 tau=2.0, v_threshold=1.0, v_reset=0.0, alpha=2.0,
                 detach_reset=True, backend="torch"):
        super().__init__()

        if activation not in ("lif", "relu"):
            raise ValueError(f"activation must be 'lif' or 'relu', got {activation!r}")
        if backend not in ("torch", "cupy"):
            raise ValueError(f"backend must be 'torch' or 'cupy', got {backend!r}")
        if num_timesteps < 1:
            raise ValueError("num_timesteps must be >= 1")

        self.num_classes     = num_classes
        self.encoding_type   = encoding_type
        self.activation      = activation
        self.normalize_input = normalize_input
        self.tau             = tau
        self.v_threshold     = v_threshold
        self.v_reset         = v_reset
        self.alpha           = alpha
        self.detach_reset    = detach_reset
        self.backend         = backend

        if activation == "relu":
            if num_timesteps != 1:
                print("[SEWResNet18] activation='relu': T forced to 1 "
                      "(a constant input replicated over T gives identical outputs).")
            self.num_timesteps = 1
        else:
            self.num_timesteps = num_timesteps

        # No try/except fallback on purpose: failing loudly is safer than
        # silently training a different (non-spiking) network.
        self._build_spikingjelly(num_classes)

    # ------------------------------------------------------------------ build
    def _build_spikingjelly(self, num_classes):
        """Build SEW-ResNet18 with SpikingJelly backend."""
        if self.activation == "lif":
            neuron_kwargs = dict(
                spiking_neuron=neuron.LIFNode,
                tau=self.tau,
                v_threshold=self.v_threshold,
                v_reset=self.v_reset,
                surrogate_function=surrogate.ATan(alpha=self.alpha),
                detach_reset=self.detach_reset,
            )
        else:
            neuron_kwargs = dict(spiking_neuron=_relu_factory)

        self.backbone = sew_resnet.sew_resnet18(
            pretrained=False,       # trained from scratch
            progress=False,
            cnf="ADD",
            **neuron_kwargs,
        )

        # Adapt first conv for single-channel (grayscale) input
        self.backbone.conv1 = layer.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Adapt classifier head
        self.backbone.fc = layer.Linear(
            self.backbone.fc.in_features, num_classes
        )

        # Multi-step mode: the backbone consumes tensors of shape [T, B, C, H, W]
        functional.set_step_mode(self.backbone, step_mode="m")

        if self.activation == "lif" and self.backend != "torch":
            functional.set_backend(
                self.backbone, backend=self.backend, instance=neuron.LIFNode
            )

        self.backbone_type   = ("SEW-ResNet18 (SpikingJelly)" if self.activation == "lif"
                                else "SEW-ResNet18 topology, ReLU control (SpikingJelly)")
        self.is_spikingjelly = True

    # --------------------------------------------------------------- forward
    def set_num_timesteps(self, num_timesteps):
        """Change T at run time (e.g. for T-ablations). Ignored for 'relu'."""
        if num_timesteps < 1:
            raise ValueError("num_timesteps must be >= 1")
        if self.activation == "relu":
            print("[SEWResNet18] activation='relu': T stays 1.")
            return
        self.num_timesteps = int(num_timesteps)

    def forward(self, x):
        """
        Forward pass using direct (constant) encoding.

        Input x of shape (B, C, H, W) is (optionally) min-max normalized to
        [0, 1] per sample and replicated across T timesteps before being fed to
        the network. LIF neurons generate spikes through their internal membrane
        dynamics. Output logits are obtained by averaging over timesteps.

        Reference: Fang et al. (2021a) — direct encoding for static images.
        """
        if x.ndim != 4:
            raise ValueError(f"Expected (B, C, H, W), got {tuple(x.shape)}")

        functional.reset_net(self.backbone)          # clean state at the start

        if self.normalize_input:
            # Per-sample min-max normalization to [0, 1]
            x_min = x.flatten(1).min(dim=1)[0].view(-1, 1, 1, 1)
            x_max = x.flatten(1).max(dim=1)[0].view(-1, 1, 1, 1)
            x = (x - x_min) / (x_max - x_min + 1e-8)

        # Direct encoding: replicate over T timesteps -> [T, B, C, H, W]
        spike_seq = x.unsqueeze(0).repeat(self.num_timesteps, 1, 1, 1, 1)

        out_seq = self.backbone(spike_seq)           # [T, B, num_classes]
        out     = out_seq.mean(dim=0)                # temporal average of logits

        functional.reset_net(self.backbone)          # and at the end
        return out

    # ------------------------------------------------------------------ info
    def get_architecture_info(self):
        """Return model metadata as a dictionary."""
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total_parameters'    : total,
            'trainable_parameters': trainable,
            'architecture'        : getattr(self, 'backbone_type', 'Unknown'),
            'activation'          : self.activation,
            'num_classes'         : self.num_classes,
            'num_timesteps'       : self.num_timesteps,
            'encoding_type'       : self.encoding_type,
            'normalize_input'     : self.normalize_input,
            'tau'                 : self.tau,
            'v_threshold'         : self.v_threshold,
            'v_reset'             : self.v_reset,
            'surrogate'           : f"ATan(alpha={self.alpha})",
            'detach_reset'        : self.detach_reset,
            'backend'             : self.backend,
            'is_spikingjelly'     : getattr(self, 'is_spikingjelly', False),
            'spikingjelly_version': _spikingjelly_version(),
            'torch_version'       : torch.__version__,
        }


def create_sew_resnet18(num_classes=7, num_timesteps=8, encoding_type="direct",
                        **kwargs):
    """
    Instantiate and return a SEWResNet18EmotionRecognizer.

    Args:
        num_classes   (int): Number of output classes. Default: 7.
        num_timesteps (int): Number of simulation timesteps. Default: 8.
        encoding_type (str): Encoding label for logging. Default: 'direct'.
        **kwargs           : Forwarded to SEWResNet18EmotionRecognizer
                             (activation, normalize_input, tau, backend, ...).

    Returns:
        SEWResNet18EmotionRecognizer: Model ready for training or inference.
    """
    model = SEWResNet18EmotionRecognizer(num_classes, num_timesteps,
                                         encoding_type, **kwargs)
    info = model.get_architecture_info()

    print(f"[create_sew_resnet18] Architecture  : {info['architecture']}")
    print(f"[create_sew_resnet18] Parameters    : {info['total_parameters']:,}")
    print(f"[create_sew_resnet18] Timesteps     : {info['num_timesteps']}")
    print(f"[create_sew_resnet18] Input min-max : {info['normalize_input']}")
    print(f"[create_sew_resnet18] SpikingJelly  : {info['spikingjelly_version']}")

    return model


def reset_snn_state(model):
    """Reset membrane potentials of all neurons in the model."""
    functional.reset_net(model.backbone)


# ------------------------------------------------------------ spike analysis
def get_spiking_layers(model):
    """
    Return an ordered dict {name: module} of all LIF/IF layers of the model.
    Empty for the 'relu' control.
    """
    return OrderedDict(
        (name, m) for name, m in model.backbone.named_modules()
        if isinstance(m, (neuron.LIFNode, neuron.IFNode))
    )


class SpikeStatsRecorder:
    """
    Record spike statistics of every spiking layer with forward hooks.

    Hooks are removed on exit, and no activation is kept in memory: only
    counters are accumulated. Use it around an evaluation loop:

        with SpikeStatsRecorder(model) as rec:
            for x, y in loader:
                model(x)
        stats = rec.summary(num_images=len(loader.dataset))

    Per layer, ``spikes`` is the total number of spikes over all timesteps and
    images, and ``neurons`` the number of neuron-timestep slots (numel of the
    output [T, B, C, H, W]); their ratio is the Average Firing Rate (AFR).
    """

    def __init__(self, model):
        self.model   = model
        self.layers  = get_spiking_layers(model)
        self.handles = []
        self.reset()

    def reset(self):
        self.spikes  = OrderedDict((n, 0.0) for n in self.layers)
        self.neurons = OrderedDict((n, 0)   for n in self.layers)
        self.shapes  = OrderedDict()

    def __enter__(self):
        for name, module in self.layers.items():
            def make_hook(layer_name):
                def hook(mod, inp, out):
                    if isinstance(out, torch.Tensor):
                        self.spikes[layer_name]  += out.detach().sum().item()
                        self.neurons[layer_name] += out.numel()
                        self.shapes.setdefault(layer_name, tuple(out.shape[2:]))
                return hook
            self.handles.append(module.register_forward_hook(make_hook(name)))
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self.handles:
            h.remove()
        self.handles = []
        return False

    def summary(self, num_images=None):
        """Return per-layer and global statistics as a dictionary."""
        per_layer = OrderedDict()
        for n in self.layers:
            per_layer[n] = {
                'spikes' : self.spikes[n],
                'neurons': self.neurons[n],
                'afr'    : self.spikes[n] / max(self.neurons[n], 1),
                'map_chw': self.shapes.get(n),
            }
        tot_s = sum(self.spikes.values())
        tot_n = sum(self.neurons.values())
        out = {
            'per_layer'     : per_layer,
            'total_spikes'  : tot_s,
            'total_neurons' : tot_n,
            'afr'           : tot_s / max(tot_n, 1),
            'sparsity'      : 1.0 - tot_s / max(tot_n, 1),
        }
        if num_images:
            out['spikes_per_image'] = tot_s / num_images
        return out


def attach_spike_hooks(model):
    """
    Backward-compatible helper: attach forward hooks to all LIF/IF layers.

    Returns:
        dict: {module: spike_tensor} populated after each forward pass.

    Note: prefer ``SpikeStatsRecorder`` for analysis pipelines. This helper
    keeps full activations in memory and its hooks are never removed.
    """
    spike_dict = {}

    def hook_fn(module, input, output):
        if isinstance(output, torch.Tensor):
            spike_dict[module] = output.detach()

    for module in get_spiking_layers(model).values():
        module.register_forward_hook(hook_fn)

    return spike_dict



# ------------------------------------------------------- SNN compliance check
def verify_snn_compliance(model, x=None, verbose=True):
    """
    Check that `model` is a genuine spiking network and report its known
    deviations from a strictly spike-driven network.

    Structural checks (returned in ``checks``; all must be True):
      * it contains spiking layers (LIF/IF) and no ReLU-like activation;
      * all spiking layers use multi-step mode;
      * every spiking layer outputs strictly binary values {0, 1};
      * the output depends on the number of time steps T (temporal dynamics).

    Informational items (returned in ``info``):
      * neuron hyper-parameters actually present in the layers;
      * the set of values seen at the input of the synaptic layers (with the
        SEW-ADD connect function it is {0, 1, 2}: integer spike counts);
      * layers that are NOT spike-driven (analog stem input, classifier).

    Args:
        x: optional batch (B, 1, H, W). With an untrained network and random
           input the model is usually silent; in that case the binary and T
           checks are repeated on a copy whose firing threshold is lowered.
    """
    if getattr(model, 'activation', 'lif') != 'lif':
        raise ValueError("verify_snn_compliance expects an activation='lif' model")

    was_training = model.training
    model.eval()
    dev = next(model.parameters()).device
    if x is None:
        torch.manual_seed(0)
        x = torch.randn(4, 1, 48, 48)
    x = x.to(dev)

    spiking = get_spiking_layers(model)
    checks, info = OrderedDict(), OrderedDict()

    # 1) presence of spiking layers, absence of non-spiking activations
    checks['has spiking (LIF/IF) layers'] = len(spiking) > 0
    relu_like = (nn.ReLU, nn.ReLU6, nn.LeakyReLU, nn.GELU, nn.SiLU, nn.Sigmoid, nn.Tanh, nn.ELU)
    offenders = [n for n, m in model.backbone.named_modules() if isinstance(m, relu_like)]
    checks['no ReLU-like activation in backbone'] = len(offenders) == 0
    info['non-spiking activations found'] = offenders

    # 2) neuron hyper-parameters actually in the layers
    params = set()
    for m in spiking.values():
        params.add((type(m).__name__, getattr(m, 'tau', None), getattr(m, 'v_threshold', None),
                    getattr(m, 'v_reset', None), getattr(m, 'detach_reset', None),
                    getattr(m, 'step_mode', None), getattr(m, 'decay_input', None)))
    info['neuron params (type, tau, V_th, V_reset, detach_reset, step_mode, decay_input)'] = sorted(
        params, key=str)
    checks['all spiking layers in multi-step mode'] = all(
        getattr(m, 'step_mode', None) == 'm' for m in spiking.values())

    # 3) binary outputs of every spiking layer, and values at synaptic inputs
    def run_binary_check(net, inp):
        out_vals, in_vals, hooks = {}, set(), []
        for n, m in get_spiking_layers(net).items():
            hooks.append(m.register_forward_hook(
                lambda mod, i, o, n=n: out_vals.setdefault(n, set()).update(
                    torch.unique(o.detach()).cpu().tolist())))
        for n, m in net.backbone.named_modules():
            if isinstance(m, (nn.Conv2d,)) and n != 'conv1':
                hooks.append(m.register_forward_hook(
                    lambda mod, i, o: in_vals.update(torch.unique(i[0].detach()).cpu().tolist())))
        with torch.no_grad():
            out = net(inp)
        for h in hooks:
            h.remove()
        binary = all(v <= {0.0, 1.0} for v in out_vals.values())
        fired = sum(1 for v in out_vals.values() if 1.0 in v)
        return binary, fired, sorted(in_vals), out

    binary, fired, in_vals, out_ref = run_binary_check(model, x)
    test_net, forced = model, False
    if fired == 0:                      # silent network (typical before training)
        test_net = copy.deepcopy(model).eval()
        for m in get_spiking_layers(test_net).values():
            m.v_threshold = 0.02
        binary, fired, in_vals, _ = run_binary_check(test_net, x)
        forced = True
    checks['spiking layers output strictly binary {0,1}'] = bool(binary and fired > 0)
    info['layers that fired during the check'] = f"{fired}/{len(spiking)}" + (
        " (threshold lowered to 0.02 on a copy: the network was silent)" if forced else "")
    info['values at synaptic inputs (conv layers except stem)'] = in_vals

    # 4) temporal dynamics: the output must depend on T
    t0 = test_net.num_timesteps
    with torch.no_grad():
        test_net.set_num_timesteps(1); o1 = test_net(x)
        test_net.set_num_timesteps(max(t0, 4)); o4 = test_net(x)
        test_net.set_num_timesteps(t0)
    checks['output depends on T (temporal dynamics)'] = not torch.allclose(o1, o4, atol=1e-6)

    # 5) known deviations from a strictly spike-driven network
    info['connect function'] = "ADD -> block outputs are integer spike counts {0,1,2}, not binary"
    info['analog (non spike-driven) layers'] = [
        "backbone.conv1: real-valued input (direct encoding)",
        "backbone.fc: fed by the average-pooled spikes (real values)"]
    info['input normalisation'] = ("per-sample min-max (non-local, off-chip pre-processing)"
                                   if model.normalize_input else "none")

    model.train(was_training)
    if verbose:
        print("SNN compliance check")
        for k, v in checks.items():
            print(f"  {k:48s}: {'PASS ✓' if v else 'FAIL ✗'}")
        print("Informational:")
        for k, v in info.items():
            print(f"  - {k}: {v}")
    return dict(checks=checks, info=info, all_ok=all(checks.values()))

# ------------------------------------------------------------------- tests
def test_sew_resnet(batch_size=2, timesteps=4, num_classes=7):
    """
    Validate SEW-ResNet18: output shape, determinism in eval mode, state reset,
    ReLU control, input-normalization switch, T change and spike recording.

    Returns:
        bool: True if all tests pass.
    """
    print("Running SEW-ResNet18 validation tests...")
    results = {}
    x = torch.randn(batch_size, 1, 48, 48)

    # 1) shape (LIF)
    model = create_sew_resnet18(num_classes, timesteps).eval()
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    results['output shape']              = tuple(out1.shape) == (batch_size, num_classes)
    results['state reset (repeatable)']  = torch.allclose(out1, out2, atol=1e-6)

    # 2) spike recording
    with SpikeStatsRecorder(model) as rec, torch.no_grad():
        model(x)
    s = rec.summary(num_images=batch_size)
    results['20 spiking layers recorded'] = len(s['per_layer']) == 20
    results['AFR in [0, 1]']              = 0.0 <= s['afr'] <= 1.0
    print(f"  (untrained network: AFR = {s['afr']:.4f}; ~0 is normal before training)")
    results['hooks removed']              = len(rec.handles) == 0

    # 3) change of T
    model.set_num_timesteps(2)
    with torch.no_grad():
        out_t2 = model(x)
    results['set_num_timesteps']          = (model.num_timesteps == 2
                                             and tuple(out_t2.shape) == (batch_size, num_classes))

    # 4) normalization switch: inspect the tensor actually fed to the backbone.
    #    (Comparing outputs is not reliable: an untrained SNN with default init is
    #    silent, so its output only depends on the classifier bias.)
    def _captured_input(normalize):
        m = create_sew_resnet18(num_classes, timesteps, normalize_input=normalize).eval()
        cap = {}
        h = m.backbone.register_forward_pre_hook(
            lambda mod, inp: cap.__setitem__('x', inp[0].detach()))
        with torch.no_grad():
            m(x)
        h.remove()
        return cap['x']                                   # [T, B, C, H, W]

    seq_on, seq_off = _captured_input(True), _captured_input(False)
    flat = seq_on[0].flatten(1)
    results['normalize_input=True -> [0,1] per sample'] = (
        torch.allclose(flat.min(dim=1)[0], torch.zeros(batch_size), atol=1e-5)
        and torch.allclose(flat.max(dim=1)[0], torch.ones(batch_size), atol=1e-5))
    results['normalize_input=False -> raw input'] = torch.equal(seq_off[0], x)
    results['input replicated over T'] = all(torch.equal(seq_on[0], seq_on[t])
                                             for t in range(1, seq_on.shape[0]))

    # 5) ReLU control
    m_r = create_sew_resnet18(num_classes, timesteps, activation='relu').eval()
    with torch.no_grad():
        out_r = m_r(x)
    results['ReLU control (T=1, no spiking layers)'] = (
        m_r.num_timesteps == 1 and len(get_spiking_layers(m_r)) == 0
        and tuple(out_r.shape) == (batch_size, num_classes))

    for name, ok in results.items():
        print(f"  {name:45s}: {'PASS ✓' if ok else 'FAIL ✗'}")

    reset_snn_state(model)
    return all(results.values())


if __name__ == "__main__":
    success = test_sew_resnet()
    print("\nSEW-ResNet18 ready." if success else "\nTest failed.")
    print()
    verify_snn_compliance(create_sew_resnet18(7, 4))