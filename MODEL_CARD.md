# Model Card - Medical Text Triage

## Finalidade

Classificador acadêmico de textos sintéticos em três níveis de urgência: `normal`, `attention` e
`urgent`. Não foi criado, validado ou autorizado para uso clínico.

## Dados

- 2.400 exemplos sintéticos e balanceados;
- textos em português gerados por templates determinísticos;
- nenhuma informação pessoal ou prontuário real;
- split estratificado 80/20 com semente 42.

## Modelo

- vetor TF-IDF com unigramas e bigramas;
- Regressão Logística com pesos balanceados;
- classificador convertido para ONNX, mantendo o mesmo vetor de entrada;
- mecanismo padrão da API: ONNX Runtime em CPU.

## Resultados locais

| Métrica | Valor |
|---|---:|
| Acurácia no holdout sintético | 1,00 |
| Macro F1 no holdout sintético | 1,00 |
| Latência média Scikit-Learn, lote 128 | 0,0598 ms |
| Latência média ONNX, lote 128 | 0,0127 ms |
| Speedup observado | 4,71x |

## Uso apropriado

- demonstração de MLOps, CI/CD, observabilidade e otimização;
- testes de integração em ambiente local;
- estudo do ciclo de vida de um modelo de NLP leve.

## Usos proibidos

- diagnóstico, triagem ou encaminhamento de pacientes;
- avaliação de risco clínico;
- decisões automatizadas sobre atendimento;
- processamento de dados reais sem governança, anonimização e aprovação institucional.

## Riscos e limitações

O vocabulário e os templates tornam a tarefa artificialmente separável. Não foram avaliados
drift, subgrupos demográficos, erros de digitação, linguagem regional, negação clínica, calibração,
robustez adversarial ou generalização para instituições reais. Um sistema clínico exigiria dados
representativos, revisão médica, validação prospectiva, segurança, privacidade e supervisão humana.
