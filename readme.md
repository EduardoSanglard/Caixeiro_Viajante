# Problema do Caixeiro Viajante

A Solução poderá ser apresentada individualmente:
Python (preferencialmente), ou em R, ou em Matlab, ou em C ou em Java.
Considere o seguinte problema de otimização (a escolha do número de 100 cidades foi feita simplesmente para tornar
o problema intratável. A solução ótima para este problema não é conhecida):
Suponha que um caixeiro deva partir de sua cidade, visitar clientes em outras 99 cidades diferentes, e então
retornar à sua cidade. Dadas as coordenadas das 100 cidades, descubra o percurso de menor distância que passe
uma única vez por todas as cidades e retorne à cidade de origem.
Para tornar a coisa mais interessante, sorte as coordenadas das cidades (representada por x e y, em um espaço de 100 por 100 pixels. Para o desenvolvimento, recomendo definir a seed com um valor constante, para que as execuções comecem com um mesmo conjunto de dados)

### Codificação

- Representação inteira: cada cromossomo conterá todos os números de 1 a 100 (cada número associado a uma cidade, e a ordem de aparecimento dos números no cromossomo vai indicar o percurso, sendo necessário fechar o
percurso da última para a primeira cidade).
- Detalhe: como trata-se de um percurso fechado, a origem do percurso pode ser qualquer uma das cidades, ao
menos para efeito da implementação computacional.
- número de possíveis percursos (soluções candidatas): 99! ≅ 9,33 × 10155
- função de adequação (fitness): o inverso da distância associada a cada percurso.
- solução ótima: desconhecida, em razão da impossibilidade de testar todas as soluções candidatas (único meio
existente para se garantir a obtenção da solução ótima);
- valores arbitrados pelo usuário (ou por vocês):
- tipo de mutação: sorteio de duas cidades para troca de posição
- taxa de mutação: 1%
- tipo de crossover: OX (uma espécie de crossover de um ponto, caracterizado pela junção de uma parte de um cromossomo com a parte de um outro, mas com a substituição das cidades repetidas pelas
ausentes, na sequência)
- taxa de crossover: 60%
- tipo de seleção: rank ou torneio (50% dos melhores)