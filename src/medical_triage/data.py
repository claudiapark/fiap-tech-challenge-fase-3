"""Deterministic synthetic dataset for academic pipeline demonstrations."""

import csv
import random
from pathlib import Path

LABELS = ("normal", "attention", "urgent")

TEMPLATES = {
    "normal": [
        "Exame de rotina sem alterações relevantes, sinais vitais estáveis e sem queixas agudas.",
        "Paciente assintomático, avaliação preventiva normal e acompanhamento "
        "habitual recomendado.",
        "Laudo sem evidências de processo agudo, parâmetros dentro dos limites esperados.",
        "Controle ambulatorial estável, sem dor, febre ou dificuldade respiratória.",
    ],
    "attention": [
        "Paciente com febre persistente e dor moderada, necessita avaliação médica no mesmo dia.",
        "Alteração laboratorial relevante sem instabilidade, recomenda-se "
        "investigação prioritária.",
        "Tosse prolongada com cansaço e piora recente, encaminhar para avaliação em curto prazo.",
        "Quadro de atenção com pressão elevada e cefaleia, sem sinais atuais de emergência.",
    ],
    "urgent": [
        "Dor torácica intensa, falta de ar súbita e sudorese; atendimento de emergência imediato.",
        "Rebaixamento de consciência e déficit neurológico agudo, acionar protocolo de urgência.",
        "Sangramento ativo importante com sinais de choque, intervenção emergencial necessária.",
        "Dificuldade respiratória grave e saturação crítica, encaminhar "
        "imediatamente à emergência.",
    ],
}

CONTEXT = [
    "Registro gerado para simulação acadêmica.",
    "Sem dados pessoais identificáveis.",
    "Triagem inicial sujeita à revisão profissional.",
    "Texto sintético para validação do pipeline.",
]


def generate_dataset(output_path: Path, rows: int = 2400, seed: int = 42) -> Path:
    """Generate a balanced, deterministic and non-clinical text dataset."""
    if rows < 2000:
        raise ValueError("Dataset must contain at least 2,000 rows")

    randomizer = random.Random(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[tuple[str, str]] = []

    for index in range(rows):
        label = LABELS[index % len(LABELS)]
        base = randomizer.choice(TEMPLATES[label])
        context = randomizer.choice(CONTEXT)
        records.append((f"{base} {context}", label))

    randomizer.shuffle(records)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("report_text", "urgency"))
        writer.writerows(records)
    return output_path


def load_dataset(path: Path) -> tuple[list[str], list[str]]:
    """Load report texts and urgency labels from CSV."""
    with path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < 2000:
        raise ValueError("Dataset must contain at least 2,000 samples")
    texts = [row["report_text"].strip() for row in rows]
    labels = [row["urgency"].strip() for row in rows]
    if not texts or set(labels) != set(LABELS):
        raise ValueError("Dataset must contain text and all urgency classes")
    return texts, labels
