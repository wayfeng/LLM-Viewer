from model_analyzer import get_analyzer
from hardwares import hardware_params
from model_params import get_available_models
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("model_id", type=str, help=f"choose one from {get_available_models()}")
parser.add_argument("hardware", type=str, help=f"choose one from {list(hardware_params.keys())}")
parser.add_argument("--config_file", type=str, default=None, help="config file")
parser.add_argument("--batchsize", type=int, default=1, help="batch size")
parser.add_argument("--seqlen", type=int, default=1024, help="sequence length")
parser.add_argument("--promptlen", type=int, default=128, help="prompt sequence length")
parser.add_argument("--w_bit", type=int, default=16, help="weight bitwidth")
parser.add_argument("--a_bit", type=int, default=16, help="temporary activation bitwidth")
parser.add_argument("--kv_bit", type=int, default=16, help="kv cache bitwidth")
parser.add_argument("--use_flashattention", action="store_true", help="use flash attention")
parser.add_argument("--tp-size", type=int, default=1, help="the number of devices for tensor parallelism to use")
args = parser.parse_args()

hw = args.hardware
assert hw in hardware_params
analyzer=get_analyzer(args.model_id)
ret = analyzer.analyze_generate_task(
    args.promptlen,
    args.seqlen,
    args.batchsize,
    args.w_bit,
    args.a_bit,
    args.kv_bit,
    args.use_flashattention,
    tp_size=args.tp_size,
    bandwidth=hardware_params[hw]["bandwidth"],
    fp16_tops=hardware_params[hw]["FP16"],
    int8_tops=hardware_params[hw]["INT8"],
    onchip_buffer=hardware_params[hw]["onchip_buffer"])
elapse = ret["inference_time"]
prefill_elapse = ret["prefill_time"]
print(f"{args.hardware}: 1st token latency {prefill_elapse:.2f} ms, total latency {elapse:.2f} ms, throughput {args.seqlen * args.batchsize * 1000 / elapse:.2f} Token/sec")
