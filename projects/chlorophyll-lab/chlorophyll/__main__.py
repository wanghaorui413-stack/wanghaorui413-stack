import argparse
from pathlib import Path


def _main():
    parser = argparse.ArgumentParser(description="叶绿素a真实光谱反演项目")
    parser.add_argument("--version", action="version", version="chlorophyll-lab 0.2.0")
    parser.add_argument("--home", type=Path, help="独立数据/实验工作目录；也可用CHLOROPHYLL_HOME")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("download", help="下载官方GLORIA数据并记录来源")
    prepare = commands.add_parser("prepare", help="质控、按水体分组并冻结划分")
    prepare.add_argument("--dataset", default="gloria")
    prepare.add_argument("--csv", type=Path)
    prepare.add_argument("--source-name", help="自有数据来源；确认Rrs为sr^-1、浓度为ug/L")
    prepare.add_argument("--rrs-unit", choices=["sr^-1", "water_leaving_reflectance"])
    prepare.add_argument("--target-unit", choices=["ug/L", "mg/m3", "mg/L"])
    train = commands.add_parser("train", help="仅使用训练和验证集拟合与选模")
    train.add_argument("--dataset", default="gloria")
    train.add_argument("--run", default="gloria_v1")
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--patience", type=int, default=15)
    train.add_argument("--lr", type=float, default=0.001)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--config", type=Path, help="JSON训练配置；显式命令行参数优先")
    evaluate = commands.add_parser("evaluate", help="方案冻结后进行一次独立测试")
    evaluate.add_argument("--run", default="gloria_v1")
    predict = commands.add_parser("predict", help="CSV批量浓度预测")
    predict.add_argument("--run", default="gloria_v1")
    predict.add_argument("--package", type=Path, help="导出的独立模型目录")
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument(
        "--model", choices=["selected", "cnn", "mlp", "ridge", "pls", "random_forest", "mean_log"], default="selected"
    )
    serve = commands.add_parser("serve", help="启动本机预测页面")
    serve.add_argument("--run", default="gloria_v1")
    serve.add_argument("--port", type=int, default=8766)
    serve.add_argument("--open", action="store_true")
    serve.add_argument("--package", type=Path, help="仅加载独立发布模型包")
    export = commands.add_parser("export", help="导出仅含选定模型的独立推理包")
    export.add_argument("--run", default="gloria_v1")
    export.add_argument("--output", type=Path, required=True)
    rebuild = commands.add_parser("rebuild-report", help="从冻结预测重建报告，不重新评估选模")
    rebuild.add_argument("--run", default="gloria_v1")
    satellite_model = commands.add_parser("satellite-model", help="按真实RSR卷积GLORIA，训练独立Sentinel-2模型")
    satellite_model.add_argument("--sensor", choices=["S2A_MSI", "S2B_MSI"], default="S2A_MSI")
    satellite_model.add_argument("--dataset", default="gloria_v2")
    satellite_model.add_argument("--output", type=Path)
    satellite = commands.add_parser("satellite", help="Sentinel-2 L1C校正及独立七波段栅格反演")
    operations = satellite.add_subparsers(dest="operation", required=True)
    inspect = operations.add_parser("inspect", help="核查已校正ACOLITE产品的物理量和网格")
    inspect.add_argument("--input", type=Path, required=True)
    correct = operations.add_parser("correct", help="调用官方ACOLITE处理真实L1C")
    correct.add_argument("--processor", type=Path, required=True)
    correct.add_argument("--input", type=Path, required=True)
    correct.add_argument("--output", type=Path, required=True)
    correct.add_argument("--limit", nargs=4, type=float, required=True, metavar=("SOUTH", "WEST", "NORTH", "EAST"))
    raster = operations.add_parser("predict", help="导出20m浓度GeoTIFF与质量掩膜")
    raster.add_argument("--input", type=Path, required=True)
    raster.add_argument("--model", type=Path, required=True)
    raster.add_argument("--output", type=Path, required=True)
    raster.add_argument("--quality", type=Path)
    args = parser.parse_args()
    if args.command == "train" and args.config:
        import json, sys

        path = args.config
        config = json.loads(path.read_text(encoding="utf-8-sig"))
        allowed = {
            "dataset": str,
            "run": str,
            "epochs": int,
            "patience": int,
            "lr": (int, float),
            "batch_size": int,
            "seed": int,
        }
        if not isinstance(config, dict) or set(config) - set(allowed):
            raise ValueError("训练配置存在未知字段")
        for key, value in config.items():
            if isinstance(value, bool) or not isinstance(value, allowed[key]):
                raise ValueError(f"配置类型错误：{key}")
            flag = "--" + key.replace("_", "-")
            if not any(a == flag or a.startswith(flag + "=") for a in sys.argv[1:]):
                setattr(args, key, value)
        args.config = str(path.resolve())
    if args.home:
        import os

        os.environ["CHLOROPHYLL_HOME"] = str(args.home.resolve())
        args.home = str(args.home.resolve())
    if args.command == "download":
        from .download import download

        download()
    elif args.command == "prepare":
        from .data import prepare as prepare_data

        if args.csv and (not args.rrs_unit or not args.target_unit):
            raise ValueError("自有CSV须显式声明--rrs-unit和--target-unit；不能从数值范围推断单位")
        contract = {"rrs_unit": args.rrs_unit, "target_unit": args.target_unit} if args.csv else None
        prepare_data(args.dataset, args.csv, args.source_name, contract)
    elif args.command == "train":
        from .train import train as train_models

        train_models(args)
    elif args.command == "evaluate":
        from .evaluate import evaluate as evaluate_run

        evaluate_run(args.run)
    elif args.command == "predict":
        from .inference import predict_file

        predict_file(args.run, args.input, args.output, args.model, args.package)
    elif args.command == "serve":
        from .serve import serve as serve_run

        serve_run(args.run, args.port, args.open, package_path=args.package)
    elif args.command == "export":
        from .release import export_model

        print(export_model(args.run, args.output))
    elif args.command == "rebuild-report":
        from .evaluate import rebuild_report

        rebuild_report(args.run)
    elif args.command == "satellite-model":
        from .satellite_model import train_satellite_model

        print(train_satellite_model(args.sensor, args.dataset, args.output))
    elif args.command == "satellite":
        from .satellite import inspect_product, correct, predict_raster

        if args.operation == "inspect":
            print(inspect_product(args.input))
        elif args.operation == "correct":
            print(correct(args.processor, args.input, args.output, args.limit))
        else:
            print(predict_raster(args.input, args.model, args.output, args.quality))


def main():
    try:
        _main()
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        import json, sys

        print(
            json.dumps({"error": {"code": type(error).__name__, "message": str(error)}}, ensure_ascii=False),
            file=sys.stderr,
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
