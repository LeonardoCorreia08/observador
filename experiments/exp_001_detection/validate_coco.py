
import json

from src.utils.config_loader import load_config, resolve_path


def main():
    # Carrega as configurações centrais
    config = load_config()

    dataset_config = config["dataset"]
    experiment_config = config["experiments"]["exp_001"]

    # Resolve os caminhos a partir da raiz do projeto
    dataset_root = resolve_path(dataset_config["root"])

    image_dir = dataset_root / dataset_config["image_dir"]

    annotation_file = (
        dataset_root
        / dataset_config["annotation_dir"]
        / dataset_config["annotation_file"]
    )

    output_dir = resolve_path(experiment_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print(f"{config['project']['name']} | exp_001")
    print(f"Experimento: {experiment_config['name']}")
    print("=" * 55)

    # Validação dos diretórios
    if not image_dir.exists():
        raise FileNotFoundError(
            f"Diretório de imagens não encontrado: {image_dir}"
        )

    if not annotation_file.exists():
        raise FileNotFoundError(
            f"Arquivo de anotações não encontrado: {annotation_file}"
        )

    # Carregamento das imagens
    images = list(image_dir.glob("*.jpg"))

    # Carregamento das anotações COCO
    with open(annotation_file, "r", encoding="utf-8") as file:
        annotations = json.load(file)

    # Coleta dos dados
    image_count = len(images)
    annotated_image_count = len(annotations["images"])
    object_count = len(annotations["annotations"])
    categories = annotations["categories"]

    # Exibição dos resultados
    print("\nRESULTADOS DA VALIDAÇÃO")
    print("-" * 55)

    print(f"Imagens encontradas: {image_count}")
    print(f"Imagens anotadas: {annotated_image_count}")
    print(f"Anotações de objetos: {object_count}")
    print(f"Categorias: {len(categories)}")

    print("\nCATEGORIAS DISPONÍVEIS")

    for category in categories:
        print(
            f"ID: {category['id']} | "
            f"Nome: {category['name']}"
        )

    # Relatório do experimento
    report = {
        "experiment": "exp_001",
        "experiment_name": experiment_config["name"],
        "dataset": dataset_config["name"],
        "images_found": image_count,
        "annotated_images": annotated_image_count,
        "object_annotations": object_count,
        "categories_count": len(categories),
        "validation_status": (
            "SUCCESS"
            if image_count == annotated_image_count
            else "WARNING"
        )
    }

    report_file = output_dir / "validation_report.json"

    with open(report_file, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)

    print("\n" + "=" * 55)
    print(f"Status: {report['validation_status']}")
    print(f"Relatório salvo em: {report_file}")
    print("=" * 55)


if __name__ == "__main__":
    main()
