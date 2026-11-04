# Roteiro do vídeo STAR - até 5 minutos

## 0:00-0:40 - Situation

Mostre o início do README e explique que um hospital precisa priorizar laudos rapidamente, sem
substituir a decisão clínica. O projeto usa somente textos sintéticos.

## 0:40-1:05 - Task

Apresente os requisitos: API de baixa latência, CI/CD, retreino Airflow, Docker Compose,
Prometheus, Grafana e otimização ONNX.

## 1:05-2:00 - Action: arquitetura e modelo

Mostre o diagrama do README, a decisão por inferência real-time em ECS Fargate e o pipeline
TF-IDF + Regressão Logística. Explique a separação do retreino batch.

## 2:00-2:45 - Action: CI/CD e Airflow

Mostre o workflow verde e a DAG `medical_triage_retraining`, destacando ingestão, treino e
quality gate de macro F1.

## 2:45-3:45 - Action: API e observabilidade

Faça uma chamada em `/docs`, abra Prometheus e o dashboard Grafana. Mostre total de chamadas,
latência p95 e taxa de erros.

## 3:45-4:25 - Result

Mostre a comparação de latência. No ensaio local, o Scikit-Learn teve média de 0,0598 ms e o
ONNX 0,0127 ms para lote de 128, aproximadamente 4,71x de speedup.

## 4:25-4:55 - Limitações

Explique que o F1 perfeito decorre dos templates sintéticos e não indica desempenho clínico.
Mencione necessidade de dados representativos, validação prospectiva, privacidade e supervisão
humana. Encerre antes de cinco minutos.
