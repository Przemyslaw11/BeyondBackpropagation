import unittest
from types import SimpleNamespace
from unittest.mock import patch

from beyond_backprop.runtime import backend_policy


class FakeTorch:
    def __init__(
        self, cuda_available: bool, mps_available: bool, mps_built: bool
    ) -> None:
        self.cuda = SimpleNamespace(is_available=lambda: cuda_available)
        self.backends = SimpleNamespace(
            mps=SimpleNamespace(
                is_available=lambda: mps_available,
                is_built=lambda: mps_built,
            )
        )

    @staticmethod
    def device(name: str) -> SimpleNamespace:
        return SimpleNamespace(type=name)


class BackendPolicyTests(unittest.TestCase):
    def test_local_backend_prefers_mps_for_auto(self) -> None:
        config = {"general": {"backend": "local"}}
        backend = backend_policy.get_execution_backend(config)

        with patch.object(
            backend_policy,
            "torch",
            FakeTorch(cuda_available=False, mps_available=True, mps_built=True),
        ):
            device = backend.resolve_device("auto")

        self.assertEqual(device.type, "mps")

    def test_local_backend_uses_cpu_when_mps_unavailable(self) -> None:
        config = {"general": {"backend": "local"}}
        backend = backend_policy.get_execution_backend(config)

        with patch.object(
            backend_policy,
            "torch",
            FakeTorch(cuda_available=False, mps_available=False, mps_built=False),
        ):
            device = backend.resolve_device("auto")

        self.assertEqual(device.type, "cpu")

    def test_local_backend_enables_mps_fallback_env(self) -> None:
        config = {"general": {"backend": "local"}}
        backend = backend_policy.get_execution_backend(config)

        overrides = backend.prepare_environment()
        self.assertEqual(overrides.get("PYTORCH_ENABLE_MPS_FALLBACK"), "1")

    def test_local_backend_dataloader_defaults(self) -> None:
        config = {"general": {"backend": "local"}}
        backend = backend_policy.get_execution_backend(config)

        defaults = backend.resolve_dataloader_defaults()
        self.assertEqual(defaults["num_workers"], 0)
        self.assertFalse(defaults["pin_memory"])

    def test_slurm_backend_defaults_to_slurm(self) -> None:
        config = {"general": {}}
        backend = backend_policy.get_execution_backend(config)
        self.assertEqual(backend.name, "slurm")

    def test_slurm_backend_fallback_device_no_cuda(self) -> None:
        config = {"general": {"backend": "slurm"}}
        backend = backend_policy.get_execution_backend(config)

        with patch.object(
            backend_policy,
            "torch",
            FakeTorch(cuda_available=False, mps_available=False, mps_built=False),
        ):
            device = backend.resolve_device("auto")

        self.assertEqual(device.type, "cpu")

    def test_slurm_backend_dataloader_defaults_without_cuda(self) -> None:
        config = {"general": {"backend": "slurm"}}
        backend = backend_policy.get_execution_backend(config)

        with patch.object(
            backend_policy,
            "torch",
            FakeTorch(cuda_available=False, mps_available=False, mps_built=False),
        ):
            defaults = backend.resolve_dataloader_defaults()

        self.assertEqual(defaults["num_workers"], 0)
        self.assertFalse(defaults["pin_memory"])

    def test_dataloader_defaults_honors_legacy_config(self) -> None:
        config = {
            "general": {"backend": "local"},
            "data_loader": {"num_workers": 2, "pin_memory": False},
        }
        backend = backend_policy.get_execution_backend(config)

        with patch.object(
            backend_policy,
            "torch",
            FakeTorch(cuda_available=False, mps_available=False, mps_built=False),
        ):
            defaults = backend.resolve_dataloader_defaults(config)

        self.assertEqual(defaults["num_workers"], 2)
        self.assertFalse(defaults["pin_memory"])

    def test_dataloader_defaults_honors_backend_config(self) -> None:
        config = {
            "general": {"backend": "local"},
            "backend": {
                "local": {"data_loader": {"num_workers": 1, "pin_memory": False}}
            },
        }
        backend = backend_policy.get_execution_backend(config)

        with patch.object(
            backend_policy,
            "torch",
            FakeTorch(cuda_available=False, mps_available=False, mps_built=False),
        ):
            defaults = backend.resolve_dataloader_defaults(config)

        self.assertEqual(defaults["num_workers"], 1)
        self.assertFalse(defaults["pin_memory"])

    def test_backend_config_takes_precedence_over_legacy(self) -> None:
        config = {
            "general": {"backend": "local"},
            "data_loader": {"num_workers": 4, "pin_memory": True},
            "backend": {
                "local": {"data_loader": {"num_workers": 0, "pin_memory": False}}
            },
        }
        backend = backend_policy.get_execution_backend(config)

        with patch.object(
            backend_policy,
            "torch",
            FakeTorch(cuda_available=False, mps_available=False, mps_built=False),
        ):
            defaults = backend.resolve_dataloader_defaults(config)

        self.assertEqual(defaults["num_workers"], 0)
        self.assertFalse(defaults["pin_memory"])

    def test_resolve_results_dir_local(self) -> None:
        config = {"general": {"backend": "local"}}
        backend = backend_policy.get_execution_backend(config)
        self.assertEqual(backend.resolve_results_dir(config), "results/local")

    def test_resolve_results_dir_slurm(self) -> None:
        config = {"general": {"backend": "slurm"}}
        backend = backend_policy.get_execution_backend(config)
        self.assertEqual(backend.resolve_results_dir(config), "results")

    def test_resolve_results_dir_from_config(self) -> None:
        config = {
            "general": {"backend": "local"},
            "backend": {"local": {"results_dir": "custom_results"}},
        }
        backend = backend_policy.get_execution_backend(config)
        self.assertEqual(backend.resolve_results_dir(config), "custom_results")

    def test_resolve_log_file_default(self) -> None:
        config = {"general": {"backend": "local"}}
        backend = backend_policy.get_execution_backend(config)
        log_file = backend.resolve_log_file(config, "test_exp")
        self.assertIn("results/local/test_exp/test_exp_run.log", log_file)

    def test_slurm_backend_prefers_cuda(self) -> None:
        config = {"general": {"backend": "slurm"}}
        backend = backend_policy.get_execution_backend(config)

        with patch.object(
            backend_policy,
            "torch",
            FakeTorch(cuda_available=True, mps_available=False, mps_built=False),
        ):
            device = backend.resolve_device("auto")

        self.assertEqual(device.type, "cuda")

    def test_explicit_cpu_preference(self) -> None:
        config = {"general": {"backend": "local"}}
        backend = backend_policy.get_execution_backend(config)

        with patch.object(
            backend_policy,
            "torch",
            FakeTorch(cuda_available=True, mps_available=True, mps_built=True),
        ):
            device = backend.resolve_device("cpu")

        self.assertEqual(device.type, "cpu")

    def test_explicit_cuda_fallback_to_cpu(self) -> None:
        config = {"general": {"backend": "local"}}
        backend = backend_policy.get_execution_backend(config)

        with patch.object(
            backend_policy,
            "torch",
            FakeTorch(cuda_available=False, mps_available=False, mps_built=False),
        ):
            device = backend.resolve_device("cuda")

        self.assertEqual(device.type, "cpu")

    def test_get_execution_backend_unknown_name_defaults_slurm(self) -> None:
        config = {"general": {"backend": "nonexistent"}}
        backend = backend_policy.get_execution_backend(config)
        self.assertEqual(backend.name, "slurm")

    def test_get_execution_backend_no_config(self) -> None:
        backend = backend_policy.get_execution_backend({})
        self.assertEqual(backend.name, "slurm")


if __name__ == "__main__":
    unittest.main()
