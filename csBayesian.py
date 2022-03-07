# TRABALHO DE APRENDIZADO DE MÁQ. E CONHECIMENTO INCERTO
# @authors:
# EDSON CIZESKI RA 107514
# PEDRO LANDINS RA 103572
## year: 2022
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator as mle
from pgmpy.inference import VariableElimination

# le o arquivo
url = 'data.csv'
df = pd.read_csv(url)

# cria o modelo da rede bayesiana com suas arestas
model = BayesianNetwork([
    ('txLevel', 'txHit'), ('txLevel', 'txMK'),
    ('txHit', 'txHs'), ('txHs', 'txADR'),
    ('txADR', 'txKDA'), ('txKDA', 'flWinner'),
    ('txMK', 'txKDA'), ('txFlash', 'txKDA'),
    ('txFK', 'flWinner'), ('txBombe', 'flWinner'),
    ('txClutch', 'txBombe')])

pe = mle(model, df)

# calcula as probabilidades condicionais de cada atributo
lvl_cpd = pe.estimate_cpd('txLevel')
hit_cpd = pe.estimate_cpd('txHit')
hs_cpd = pe.estimate_cpd('txHs')
adr_cpd = pe.estimate_cpd('txADR')
kda_cpd = pe.estimate_cpd('txKDA')
mk_cpd = pe.estimate_cpd('txMK')
flash_cpd = pe.estimate_cpd('txFlash')
fk_cpd = pe.estimate_cpd('txFK')
bomb_cpd = pe.estimate_cpd('txBombe')
clutch_cpd = pe.estimate_cpd('txClutch')
winner_cpd = pe.estimate_cpd('flWinner')

# adiciona as probabilidades encontradas para o modelo
model.add_cpds(lvl_cpd, hit_cpd, hs_cpd, adr_cpd, kda_cpd,
               mk_cpd, flash_cpd, fk_cpd, bomb_cpd, clutch_cpd, winner_cpd)
model.check_model()


# CENARIOS DE USO
belief = VariableElimination(model)
# 1) Qual a chance de um jogador ter um desempenho ruim (KDA < 1),
# dado que seu level seja bom, seus hits, hs e ADR serem bons,
# assim como seu número de multi kills
res = belief.query(variables=['txKDA'],
                   evidence={'txHit': 1, 'txHs': 1,
                   'txHit': 1, 'txLevel': 1, 'txADR': 1, 'txMK': 1})
print(res)
# 2) Qual a chance de um jogador ter um desempenho bom (KDA > 1),
#  ganhar CLUTCHES, plantar/defusar uma quantidade boa de BOMBAS
# e ter um bom número de first kills, porém perder a partida
res = belief.query(variables=['flWinner'],
                   evidence={'txKDA': 1, 'txClutch': 1,
                   'txBombe': 1, 'txFK': 1})
print(res)
# 3)Qual a probabilidade de um jogador de LEVEL baixo, com poucos HITS,
# poucos HS, assim como poucas MULTI KILLS, além de um ADR baixo,
# com poucas FLASHES e portanto um KDA ruim, conseguir ganhar a partida
# tendo ganhado alguns CLUTCHES, pegado algumas FIRST KILLS e
# plantado/desarmado BOMBAS.

res = belief.query(variables=['flWinner'],
                   evidence={'txLevel': 0, 'txHit': 0, 'txHs': 0, 'txMK': 0,
                             'txADR': 0, 'txFlash': 0, 'txKDA': 0, 'txClutch': 1,
                             'txFK': 1, 'txBombe': 1})
print(res)
