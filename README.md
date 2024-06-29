# GAT Experiments

:construction: **UNDER CONSTRUCTION** :construction:

## O projeto

Este repositório armazena o código relacionado ao meu projeto de Mestrado.

O objetivo do projeto é implementar o algoritmo proposto em [Deep neighbor-aware embedding for node clustering in attributed graphs](https://www.sciencedirect.com/science/article/abs/pii/S0031320321004118), comparar os resultados objetidos com os que foram apresentados no paper original e propor variações no algoritmo.

### Testes e variações:

Até o presente momento, as sugestões de testes e variações podem incluir:

- Novos mecanismos para a detecção de centroids;
- Novas funções de Loss e variação de pesos;
- Variações na estrutura e parâmetros da rede;
- Novas funções para detecção das classes;
- Testes paramétricos variados;

Todos os testes e variações serão implementadas e documentadas neste repositório.

### Centroides por Detecção de Comunidades

Uma das abordagens escolhidas para teste e comparação é a detecção de comunidades via algoritmos como Fast Greedy, e posteriormente o mapeamento dos centroids via grau de conectividade. Isso substitui o KMeans como mecanismos inicial de detecção dos centroides. Os resultados dos testes devem ser compartilhados via este repositório.

### Centroides por Seed Expansion

Outro mecanismo a ser avaliado por este projeto é o mecanismo de Seed Expansion para a detecção dos centroides iniciais. Dessa forma, não seria necessária a detecção de comunidades. Acreditamos que esta abordagem pode trazer um ganho de eficiência para o algoritmo como um todo, mas não temos ainda resultados para afirmar a qualidade de tal abordagem.


## Autores:

Alan Demétrius Baria Valejo

Carlos Pereira Lopes Filho

Guilherme Henrique Messias
