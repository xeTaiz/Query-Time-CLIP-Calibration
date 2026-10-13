"""Evaluate the four-scene LERF-OVS image protocol with strict LOSO N1."""
from .image import run_dataset_cli


if __name__ == "__main__":
    run_dataset_cli("lerf_ovs")
