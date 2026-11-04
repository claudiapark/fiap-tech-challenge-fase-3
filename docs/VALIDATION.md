# Evidências de validação local

Data: 04/11/2026
Ambiente: macOS ARM64, Python 3.12, Docker Engine via Colima

## Qualidade de código

- `ruff format --check .`: aprovado;
- `ruff check .`: aprovado;
- `pytest`: 15 testes aprovados;
- cobertura total: 95,57%.

## Modelo e otimização

- dataset sintético: 2.400 linhas;
- split estratificado: 1.920 treino e 480 teste;
- macro F1: 1,00, resultado esperado pela simplicidade dos templates;
- latência média Scikit-Learn, lote de 128: 0,0598 ms;
- latência média ONNX Runtime, lote de 128: 0,0127 ms;
- speedup local: 4,71x.

## Docker Compose e observabilidade

- API: container saudável em `http://localhost:8000`;
- inferência ONNX: classe `urgent`, confiança 0,9599 no exemplo de emergência;
- Prometheus: target `medical-triage-api` com estado `up`;
- Grafana: datasource Prometheus provisionado com UID `prometheus`;
- dashboard `Medical Triage - API Observability` provisionado com cinco painéis.

## Airflow

A imagem `apache/airflow:3.1.0-python3.12` foi inicializada com banco efêmero e o comando
`airflow dags list-import-errors --output json` retornou `[]`, confirmando que a DAG não apresenta
erros de importação.

## Limite da evidência

O workflow do GitHub Actions foi validado estruturalmente, mas sua execução verde depende da
publicação deste repositório no GitHub. Os números de desempenho variam conforme hardware e carga.
