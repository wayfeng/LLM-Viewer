from flask import Flask, request
from flask_cors import CORS
from get_model_graph import get_model_graph
from model_params import get_available_models
from hardwares import get_available_hardwares, get_hardware_params
import argparse
import logging

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})


@app.route("/")
def index():
    return "backend server ready."


@app.route("/get_graph", methods=["POST"])
def get_graph():
    inference_config = request.json["inference_config"]
    nodes, edges, total_results, hardware_info = get_model_graph(
        request.json["model_id"],
        inference_config,
    )
    return {
        "nodes": nodes,
        "edges": edges,
        "total_results": total_results,
        "hardware_info": hardware_info,
    }

@app.route("/get_available", methods=["GET"])
def get_available():
    return {
        "available_hardwares": get_available_hardwares(),
        "available_model_ids": get_available_models(),
    }

@app.route("/get_hardware_params", methods=["POST"])
def get_hardware_params_route():
    hardware = request.json["hardware"]
    return get_hardware_params(hardware)

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--loglevel", type=str, default="info")
    args=parser.parse_args()

    logger_level = args.loglevel.upper()
    logging.basicConfig(level=getattr(logging, logger_level))

    host="127.0.0.1" if args.local else "0.0.0.0"
    app.run(debug=args.debug,host=host,port=args.port)
