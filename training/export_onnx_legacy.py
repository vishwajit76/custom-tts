"""Run piper.train.export_onnx with the legacy TorchScript exporter (dynamo exporter fails on torch>=2.9)."""
import functools, runpy, torch
torch.onnx.export = functools.partial(torch.onnx.export, dynamo=False)
runpy.run_module("piper.train.export_onnx", run_name="__main__")
