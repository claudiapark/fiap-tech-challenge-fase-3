# Medical Text Triage - FIAP Tech Challenge Fase 3

Pipeline acadêmico de Engenharia de Machine Learning para classificar textos de laudos em
`normal`, `attention` ou `urgent`. A entrega reúne API FastAPI, modelo Scikit-Learn, inferência
ONNX, CI/CD, DAG Airflow e monitoramento com Prometheus e Grafana.

> **Aviso de segurança:** este projeto usa dados sintéticos, não contém informações de pacientes
> e não deve ser utilizado para diagnóstico, priorização ou decisão clínica real.

## Arquitetura

```text
                          GitHub Actions
                      lint -> tests -> build
                               |
                               v
Cliente -> FastAPI -> TF-IDF -> ONNX Runtime
             |                    |
             +---- /metrics ------+
                      |
                 Prometheus -> Grafana

Airflow: geração de dados -> treino -> exportação ONNX -> quality gate
```

### Decisão de nuvem

O cenário é de **inferência em tempo real**, pois a classificação precisa responder durante a
triagem. A proposta para produção é AWS Application Load Balancer + ECS Fargate em sub-redes
privadas, com imagens no ECR, autoscaling por requisições/CPU, TLS no ALB e logs sem o conteúdo
dos laudos. Prometheus e Grafana podem migrar para Amazon Managed Service for Prometheus e
Amazon Managed Grafana. O retreino é assíncrono: uma agenda do MWAA/Airflow inicia o pipeline,
valida métricas e publica uma nova imagem somente após aprovação. Esse desenho separa o caminho
de baixa latência do processamento batch e permite rollback por versão da imagem.

## Componentes obrigatórios

- API REST FastAPI com `/health`, `/predict`, `/docs` e `/metrics`;
- modelo TF-IDF + Regressão Logística e classificador exportado para ONNX;
- comparação reproduzível de latência Scikit-Learn versus ONNX Runtime;
- workflow GitHub Actions com formatação, lint, testes, treino e build Docker;
- DAG Airflow com ingestão, treino e quality gate;
- Docker Compose com API, Prometheus e Grafana;
- dashboard provisionado com total de requisições, taxa, p95, erros e classes previstas.

## Execução local

Requisitos: `uv`, Docker e Docker Compose.

```bash
uv sync --locked
uv run python scripts/generate_dataset.py
uv run python scripts/train.py
uv run pytest
```

Inicie toda a stack:

```bash
docker compose up --build -d
docker compose ps
```

Serviços:

| Serviço | URL | Credenciais locais |
|---|---|---|
| API | <http://localhost:8000/docs> | não aplicável |
| Métricas | <http://localhost:8000/metrics> | não aplicável |
| Prometheus | <http://localhost:9090> | não aplicável |
| Grafana | <http://localhost:3000> | `admin` / `admin` |

Faça uma predição:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"report_text":"Dor torácica intensa e falta de ar súbita, emergência."}'
```

Gere tráfego para o dashboard:

```bash
for i in {1..30}; do
  curl -s -X POST http://localhost:8000/predict \
    -H 'Content-Type: application/json' \
    -d '{"report_text":"Febre persistente e dor moderada, avaliar ainda hoje."}' >/dev/null
done
```

Encerre a stack:

```bash
docker compose down
```

## Airflow

A DAG `medical_triage_retraining` executa geração/ingestão, treinamento/exportação e validação
de `macro_f1 >= 0.80`. Para subir o serviço opcional:

```bash
docker compose --profile orchestration up airflow
```

Depois, acesse <http://localhost:8080> e execute a DAG manualmente ou aguarde a agenda semanal.
As credenciais temporárias do modo standalone aparecem nos logs do contêiner.

## Modelo e otimização

O pré-processamento usa TF-IDF com unigramas e bigramas. A Regressão Logística utiliza pesos
balanceados e split estratificado com `random_state=42`. Apenas o classificador é convertido para
ONNX; o mesmo vetor TF-IDF é usado nos dois caminhos para que a comparação de latência seja justa.

Os números reproduzidos ficam em `reports/metrics.json` e podem ser exibidos com:

```bash
uv run python scripts/benchmark.py
```

Resultado local de referência em 04/11/2026, medindo lotes de 128 vetores por 150 repetições:

| Implementação | Média | p95 |
|---|---:|---:|
| Scikit-Learn | 0,0598 ms | 0,0740 ms |
| ONNX Runtime | 0,0127 ms | 0,0167 ms |

O ONNX apresentou **4,71x de speedup** nesse ensaio. A acurácia e o macro F1 foram 1,00 no
holdout sintético; esse resultado não representa desempenho clínico e reflete a separação simples
dos templates, conforme registrado nas limitações.

## Monitoramento

Métricas principais:

- `triage_requests_total`: chamadas por método, rota e status;
- `triage_request_latency_seconds`: histograma de latência;
- `triage_errors_total`: falhas de execução;
- `triage_predictions_total`: previsões por classe e mecanismo.

O texto enviado pelo usuário não aparece em labels, métricas ou logs, evitando cardinalidade alta
e exposição acidental de conteúdo sensível.

## Estrutura

```text
src/medical_triage/       aplicação, treino e inferência
dags/                     DAG de retreino Airflow
monitoring/               Prometheus e dashboard Grafana
scripts/                  geração, treino e benchmark
tests/                    testes unitários e de integração
.github/workflows/        CI/CD
```

## Limitações

- dados sintéticos e balanceados não representam distribuição clínica real;
- vocabulário pequeno pode superestimar as métricas;
- não há validação prospectiva, calibração clínica ou revisão regulatória;
- o benchmark local mede o classificador em lote e varia conforme hardware;
- autenticação, criptografia de dados e trilha de auditoria precisam ser adicionadas antes de um
  experimento controlado com dados reais anonimizados.

## Licença e responsabilidade

Uso exclusivamente acadêmico. Nenhum resultado deve ser interpretado como recomendação médica.

## Evidências

Consulte [`docs/VALIDATION.md`](docs/VALIDATION.md) para os testes locais realizados e
[`docs/VIDEO_STAR.md`](docs/VIDEO_STAR.md) para o roteiro de apresentação.
