"""Sanity check the single-skill full config's FSDP settings."""

import argparse

import jax

import openpi.training.config as _config
import openpi.training.sharding as sharding


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-count", type=int, default=None, help="Expected device count to validate against.")
    args = parser.parse_args()

    config = _config.get_config("pi05_b1k-sampled_single_skill-full")

    assert config.fsdp_devices > 1, f"Expected FSDP enabled, got fsdp_devices={config.fsdp_devices}"

    actual_device_count = jax.device_count()
    target_device_count = args.device_count or actual_device_count
    assert target_device_count > 0, "Expected at least one device"
    assert target_device_count % config.fsdp_devices == 0, (
        f"Device count {target_device_count} must be divisible by fsdp_devices={config.fsdp_devices}"
    )
    assert config.batch_size % target_device_count == 0, (
        f"Global batch {config.batch_size} must be divisible by device count {target_device_count}"
    )

    per_device_batch = config.batch_size // target_device_count

    if actual_device_count == target_device_count:
        mesh = sharding.make_mesh(config.fsdp_devices)
        mesh_shape = mesh.shape
    else:
        mesh_shape = {
            sharding.BATCH_AXIS: target_device_count // config.fsdp_devices,
            sharding.FSDP_AXIS: config.fsdp_devices,
        }

    print(f"[PASS] config name: {config.name}")
    print(f"[PASS] fsdp_devices: {config.fsdp_devices}")
    print(f"[PASS] global batch_size: {config.batch_size}")
    print(f"[PASS] target device count: {target_device_count}")
    print(f"[PASS] actual jax.device_count(): {actual_device_count}")
    print(f"[PASS] mesh shape: {mesh_shape}")
    print(f"[PASS] per-device batch: {per_device_batch}")


if __name__ == "__main__":
    main()
