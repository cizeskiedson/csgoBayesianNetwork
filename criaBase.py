# TRABALHO DE APRENDIZADO DE MÁQ. E CONHECIMENTO INCERTO
# @authors:
# EDSON CIZESKI RA 107514
# PEDRO LANDINS RA 103572
## year: 2022

# CENARIOS DE USO
# 1) Se um jogador tem o KD maior do que 1, qual é a sua taxa de vitória?
# 2) Se um jogador tem mais de X assists, qual é a sua taxa de vitória?
# 3) Se um jogador tem uma taxa de HS alta e muito dano por round, isso impacta no número de Kills?

import pandas as pd
url = 'data/tb_lobby_stats_player.csv'
df = pd.read_csv(url)

# tira os jogos que foram de menos de 16 rounds
df.drop(df[df.qtRoundsPlayed < 16].index, inplace=True)
# transformar variaveis continuas em descritivas
# ATRIBUTO HITS
# HITS / SHOTS
# escala binária: acima de 40% ou abaixo
df['result'] = df['qtHits']/df['qtShots']
df.loc[df.result < 0.40, "txHit"] = 0
df.loc[df.result >= 0.40, "txHit"] = 1
#print(df[df['txHit'] == df['txHit'].max()])

# ATRIBUTO DAMAGE
# damage / rounds
# escala: > 70 mt bom, < 70 ok
df['ADR'] = df['vlDamage'] / df['qtRoundsPlayed']
df.loc[df.ADR >= 70, "txADR"] = 1
df.loc[df.ADR < 70, "txADR"] = 0


# ATRIBUTO HEADSHOT
# escala binária: maior que metade das kills = bom(1)
df.loc[df.qtHs < (df.qtKill / 2), "txHs"] = 0
df.loc[df.qtHs >= (df.qtKill / 2), "txHs"] = 1
# print(df.txHs)

# ATRIBUTO FLASH ASSIST
# escala binária: maior que metade das assists = bom(1)
df.loc[df.qtFlashAssist < (df.qtAssist / 2), "txFlash"] = 0
df.loc[df.qtFlashAssist >= (df.qtAssist / 2), "txFlash"] = 1
# print(df.txFlash)


# ATRIBUTO KDA
# KILL +3 DEATH -2 ASSIST +1
df['KDA'] = (df['qtKill'] * 3) + (df['qtAssist']) - (df['qtDeath'] * 2)
# escala binária: positivo = bom(1)
df.loc[df.KDA < 1, "txKDA"] = 0
df.loc[df.KDA >= 1, "txKDA"] = 1

# ATRIBUTO MULTIKILL
# MULTIKILLS (2K, 3K, 4K, 5K)
# ESCALA BINARIA
# MAIOR QUE 10 : OTIMO
df['MK'] = (df['qt5Kill'] * 5) + (df['qt4Kill']
                                  * 4) + (df['qt3Kill'] * 3) + (df['qt2Kill'] *
                                                                2)

df.loc[df.MK > 10, "txMK"] = 1
df.loc[df.MK <= 10, "txMK"] = 0
# criar a escala das variaveis
# criar novo dataSet com os valores novos

# ATRIBUTO CLUTCHES
# > 1 : BOM
# 0: OK
df.loc[df.qtClutchWon >= 1, "txClutch"] = 1
df.loc[df.qtClutchWon < 1, "txClutch"] = 0

# ATRIBUTO LEVEL
# > media : BOM
# < media : RUIM
df.loc[df.vlLevel > df['vlLevel'].mean(), "txLevel"] = 1
df.loc[df.vlLevel <= df['vlLevel'].mean(), "txLevel"] = 0

# ATRIBUTO FIRST KILL
# > media : bom
# < media : ruim
df.loc[df.qtFirstKill > df['qtFirstKill'].mean(), "txFK"] = 1
df.loc[df.qtFirstKill <= df['qtFirstKill'].mean(), "txFK"] = 0

# ATRIBUTO BOMBE PLANT + DEFUSE
# > media: bom
# < media: ruim
df['BOMBE'] = df['qtBombeDefuse'] + df['qtBombePlant']
df.loc[df.BOMBE > df['BOMBE'].mean(), "txBombe"] = 1
df.loc[df.BOMBE <= df['BOMBE'].mean(), "txBombe"] = 0


# calcular a rede e as probabilidades partir do novo dataSet
df.to_csv("data.csv")
