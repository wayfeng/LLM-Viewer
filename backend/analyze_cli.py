from model_analyzer import get_analyzer
from model_params import get_available_models
from hardwares import hardware_params
from utils import str_number, str_number_time
import argparse
import time
import os

def save_csv(results, save_path=None):
    if save_path is None:
        save_path = f"output/{args.model_id[:args.model_id.rfind('/')]}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path += f"{args.model_id[args.model_id.rfind('/'):]}"

    for stage in results.keys():
        if stage == "total_results":
            continue

        file_name = f"{save_path}_{stage}.csv"

        with open(file_name, "a+") as f:
            f.write(
                f"\n\n# [{time.strftime('%Y-%m-%d %H:%M:%S')}] {args.model_id} {args.hardware} w_bit={args.w_bit} a_bit={args.a_bit} kv_bit={args.kv_bit} batchsize={args.batchsize} seqlen={args.seqlen} tp_size={args.tp_size}\n"
            )
            # legend
            f.write(
                f"layer_name,OPs,Access,arithmetic_intensity,performance,bound,load_weight,load_act,store_act,load_kv_cache,store_kv_cache,inference_time\n"
            )
            for layer_name, result in results[stage].items():
                f.write(
                    f"{layer_name},{str_number(result['OPs'])},{str_number(result['memory_access'])}B,{str_number(result['arithmetic_intensity'])},{str_number(result['performance'])},"
                    f"{result['bound']},{str_number(result['load_weight'])}B,{str_number(result['load_act'])}B,{str_number(result['store_act'])}B,{str_number(result['load_kv_cache'])}B,"
                    f"{str_number(result['store_kv_cache'])}B,{str_number_time(result['inference_time'])}s\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id", type=str, help=f"choose one from {get_available_models()}")
    parser.add_argument("hardware", type=str, help=f"choose one from {list(hardware_params.keys())}")
    parser.add_argument("--config_file", type=str, default=None, help="config file")
    parser.add_argument("-b", "--batchsize", type=int, default=1, help="batch size")
    parser.add_argument("-W", "--image_width", type=int, default=1024, help="input image width for vision models")
    parser.add_argument("-H", "--image_height", type=int, default=1024, help="input image height for vision models")
    parser.add_argument("--seqlen", type=int, default=1024, help="sequence length")
    parser.add_argument("--genlen", type=int, default=0, help="generated sequence length for chat stage")
    parser.add_argument("--w_bit", type=int, default=16, help="weight bitwidth")
    parser.add_argument("--a_bit", type=int, default=16, help="temporary activation bitwidth")
    parser.add_argument("--kv_bit", type=int, default=16, help="kv cache bitwidth")
    parser.add_argument("--use_flashattention", action="store_true", help="use flash attention")
    parser.add_argument("--tp-size", type=int, default=1, help="the number of devices for tensor parallelism to use")
    args = parser.parse_args()

    hw = args.hardware
    assert hw in hardware_params

    if args.genlen > 0:
         print(f"analyzing chat stage with genlen={args.genlen}...")
         stage = 'chat'
    else:
        print(f"analyzing prefill stage with seqlen={args.seqlen}...")
        stage = 'prefill'
    analyzer = get_analyzer(args.model_id)
    configs = {
        'stage': stage,
        "batchsize": args.batchsize,
        "seqlen": args.seqlen,
        "w_bit": args.w_bit,
        "a_bit": args.a_bit,
        "kv_bit": args.kv_bit,
        "use_flashattention": args.use_flashattention,
        "tp_size": args.tp_size,
        "bandwidth": hardware_params[hw]["bandwidth"],
        "fp16_tops": hardware_params[hw]["FP16"],
        "int8_tops": hardware_params[hw]["INT8"],
        "onchip_buffer": hardware_params[hw]["onchip_buffer"],
        "image_size": (args.image_width, args.image_height),
    }
    results = analyzer.analyze(**configs)
    save_csv(results)
