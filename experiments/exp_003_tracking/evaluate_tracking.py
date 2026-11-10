"""
Observador EXP-003: avaliador MOT17 com TrackEval.
Execute na raiz: python -m experiments.exp_003_tracking.evaluate_tracking
Dependência: pip install trackeval
"""
from __future__ import annotations
import configparser, csv, importlib.util, json, subprocess, sys
from collections import Counter
from pathlib import Path
from typing import Any

def load_config_dict() -> dict[str, Any]:
    from src.utils.config_loader import load_config
    cfg = load_config()
    if not isinstance(cfg, dict):
        raise TypeError("load_config() deve retornar dict.")
    return cfg


def get_root(cfg: dict[str, Any]) -> Path:
    repo = Path(__file__).resolve().parents[2]
    value = Path(cfg["project"]["root"])
    return value.resolve() if value.is_absolute() else (repo / value).resolve()


def resolve(root: Path, value: str) -> Path:
    p = Path(value)
    return p.resolve() if p.is_absolute() else (root / p).resolve()


def read_gt(path: Path, limit: int):
    if not path.is_file():
        raise FileNotFoundError(f"GT não encontrado: {path}")
    rows, per_frame, per_id = [], Counter(), Counter()
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row:
                continue
            if len(row) < 9:
                raise ValueError(f"Linha GT inválida: {row}")
            frame = int(float(row[0]))
            if not 1 <= frame <= limit:
                continue
            # Preservar todas as classes/flags para o pré-processamento MOT17.
            rows.append(row)
            if row[6] == "1" and row[7] == "1":
                per_frame[frame] += 1
                per_id[row[1]] += 1
    return rows, per_frame, per_id


def read_predictions(path: Path, limit: int, classes: list[int] | None):
    if not path.is_file():
        raise FileNotFoundError(f"CSV de predições não encontrado: {path}")
    rows, per_frame, per_id = [], Counter(), Counter()
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"frame","track_id","class_id","confidence","x1","y1","x2","y2"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Colunas ausentes: {sorted(missing)}")
        for r in reader:
            frame, cls = int(float(r["frame"])), int(float(r["class_id"]))
            if not 1 <= frame <= limit or (classes and cls not in classes):
                continue
            x1,y1,x2,y2 = (float(r[k]) for k in ("x1","y1","x2","y2"))
            w,h = x2-x1,y2-y1
            if w <= 0 or h <= 0:
                continue
            tid = int(float(r["track_id"]))
            rows.append({"frame":frame,"id":tid,"x":x1,"y":y1,"w":w,"h":h,
                         "conf":float(r["confidence"])})
            per_frame[frame] += 1
            per_id[tid] += 1
    return rows, per_frame, per_id


def write_diagnostics(out: Path, limit: int, gt_f: Counter, pr_f: Counter,
                      gt_id: Counter, pr_id: Counter):
    out.mkdir(parents=True, exist_ok=True)
    with (out/"metrics_by_frame.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame","gt_valid_pedestrians","predictions","pred_minus_gt"])
        for frame in range(1,limit+1):
            a,b=gt_f[frame],pr_f[frame]
            w.writerow([frame,a,b,b-a])
    with (out/"metrics_by_identity.csv").open("w", newline="", encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["source","identity_id","observations"])
        for k,v in sorted(gt_id.items(), key=lambda x:int(x[0])): w.writerow(["ground_truth",k,v])
        for k,v in sorted(pr_id.items()): w.writerow(["prediction",k,v])
    total_gt,total_pr=sum(gt_f.values()),sum(pr_f.values())
    summary={
        "frames_evaluated":limit,
        "ground_truth_valid_pedestrian_annotations":total_gt,
        "ground_truth_unique_pedestrian_ids":len(gt_id),
        "prediction_count":total_pr,
        "prediction_unique_track_ids":len(pr_id),
        "ground_truth_average_per_frame":round(total_gt/limit,4),
        "prediction_average_per_frame":round(total_pr/limit,4),
        "frames_without_predictions":sum(pr_f[i]==0 for i in range(1,limit+1)),
        "note":"Contagens diagnósticas, não TP/FP/FN. Métricas calculadas pelo TrackEval."
    }
    (out/"diagnostic_summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    return summary


def trackeval_cli() -> Path:
    spec=importlib.util.find_spec("trackeval")
    if spec is None or spec.origin is None:
        raise RuntimeError("TrackEval não instalado. Execute: pip install trackeval")
    path=Path(spec.origin).resolve().parent/"scripts"/"run_mot_challenge.py"
    if not path.is_file():
        raise FileNotFoundError(f"CLI oficial do TrackEval não encontrado: {path}")
    return path


def prepare_trackeval(out: Path, seq: str, tracker: str, limit: int,
                      gt_rows: list[list[str]], predictions: list[dict],
                      seqinfo: Path, benchmark: str):
    
    # Prepara caminhos dinâmicos baseados no root e config
    base = out / "trackeval"
    gt_root = base / "gt"
    tracker_root = base / "trackers"
    output_root = base / "output"
    
    # O TrackEval exige o formato "{benchmark}-train", ex: "MOT17-train"
    split_name = f"{benchmark}-train"
    
    # Configura pastas de GT
    seq_dir = gt_root / split_name / seq
    gt_dir = seq_dir / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)
    
    # Grava gt.txt dinamicamente
    with (gt_dir / "gt.txt").open("w", encoding="utf-8", newline="") as f:
        for row in gt_rows: f.write(",".join(row)+"\n")
            
    # Lê e copia o seqinfo.ini original da pasta RAW
    if not seqinfo.is_file(): raise FileNotFoundError(f"seqinfo.ini ausente: {seqinfo}")
    parser = configparser.ConfigParser()
    parser.read(seqinfo, encoding="utf-8")
    if not parser.has_section("Sequence"): raise ValueError("seqinfo.ini sem [Sequence]")
    parser["Sequence"]["name"] = seq
    parser["Sequence"]["seqLength"] = str(limit)
    with (seq_dir / "seqinfo.ini").open("w", encoding="utf-8") as f: parser.write(f)
        
    # Grava o resultado das predições do tracker na pasta certa
    data_dir = tracker_root / split_name / tracker / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    with (data_dir / f"{seq}.txt").open("w", encoding="utf-8", newline="") as f:
        for p in predictions:
            vals = [p["frame"], p["id"], p["x"], p["y"], p["w"], p["h"], p["conf"], -1, -1, -1]
            f.write(",".join(map(str, vals)) + "\n")
            
    # Gera o seqmap obrigatório
    seqmap = base / "sequence_map.txt"
    seqmap.write_text(f"name\n{seq}\n", encoding="utf-8")
    
    return gt_root, tracker_root, output_root, seqmap


def main():
    cfg = load_config_dict()
    root = get_root(cfg)
    ds = cfg["datasets"]["mot17"]
    tracking = cfg["tracking"]
    ev = cfg["evaluation"]["exp_003"]
    
    limit = int(ev["max_frames"])
    seq = str(ev["sequence_name"])
    tracker = str(ev["tracker_name"])
    benchmark = str(ev.get("benchmark", "MOT17"))
    
    # Resolve dinamicamente os caminhos que estão no config.yaml (relativos ao projeto)
    seq_dir = resolve(root, ev["ground_truth_sequence"])
    gt_path = seq_dir / str(ds["annotation_dir"]) / str(ds["annotation_file"])
    seqinfo = seq_dir / str(ds["sequence_info"])
    pred_path = resolve(root, ev["prediction_file"])
    out = resolve(root, ev["output_dir"])
    
    gt_rows, gt_f, gt_id = read_gt(gt_path, limit)
    preds, pr_f, pr_id = read_predictions(pred_path, limit, tracking.get("classes"))
    
    diagnostic = write_diagnostics(out, limit, gt_f, pr_f, gt_id, pr_id)
    
    # Chama a função que monta toda a subpasta e copia os arquivos RAW pra evaluation dinamicamente
    gt_root, tr_root, out_root, seqmap = prepare_trackeval(
        out, seq, tracker, limit, gt_rows, preds, seqinfo, benchmark
    )
    
    metrics = [str(x) for x in ev.get("metrics", ["HOTA", "CLEAR", "Identity"])]
    command = [
        sys.executable, str(trackeval_cli()),
        "--GT_FOLDER", str(gt_root), 
        "--TRACKERS_FOLDER", str(tr_root),
        "--TRACKERS_TO_EVAL", tracker, 
        "--TRACKER_SUB_FOLDER", "data",
        "--BENCHMARK", benchmark, 
        "--SPLIT_TO_EVAL", "train",
        "--SEQMAP_FILE", str(seqmap), 
        "--DO_PREPROC", str(bool(ev.get("do_preproc", True))),
        "--METRICS", *metrics, 
        "--OUTPUT_FOLDER", str(out_root)
    ]
    
    print("\nOBSERVADOR | EXP-003 | AVALIAÇÃO")
    print(f"Sequência: {seq} | Frames: 1–{limit}")
    print(f"GT válido: {diagnostic['ground_truth_valid_pedestrian_annotations']}")
    print(f"Predições: {diagnostic['prediction_count']}")
    print(f"IDs reais: {diagnostic['ground_truth_unique_pedestrian_ids']}")
    print(f"IDs previstos: {diagnostic['prediction_unique_track_ids']}")
    
    subprocess.run(command, cwd=root, check=True)
    
    final = {**diagnostic, "trackeval": {
        "status": "completed", 
        "benchmark": benchmark,
        "sequence": seq, 
        "metrics_requested": metrics, 
        "preprocessing_enabled": bool(ev.get("do_preproc", True)),
        "output_dir": str(out_root)
    }}
    
    (out / "evaluation_summary.json").write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Avaliação concluída. Resultados:", out_root)


if __name__ == "__main__":
    main()