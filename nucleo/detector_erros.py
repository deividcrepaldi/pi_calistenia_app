"""
Detecção do erro postural específico a partir dos ângulos reais calculados.
Aplicado apenas quando o modelo classifica a postura como incorreta.
A regra com maior desvio do limiar é a reportada.
"""

from typing import Optional, Tuple

from config.exercicios import Exercicio


def identificar_erro(
    exercicio: Exercicio, angulos: dict
) -> Tuple[Optional[str], Optional[str], Optional[float]]:
    """
    Avalia todas as regras biomecânicas do exercício e devolve uma tupla:
        (nome_do_erro, instrucao_correcao, desvio_em_graus)
    referente à regra de maior desvio. `desvio_em_graus` é o valor
    devolvido por `RegraErro.verificar`, que por convenção é "quantos
    graus além do limite biomecânico aceitável" — útil pra mostrar ao
    usuário a severidade do erro.

    Se nenhuma regra ativar, devolve (None, None, None).
    """
    ativadas = []
    for regra in exercicio.regras_erro:
        desvio = regra.verificar(angulos)
        if desvio is not None and desvio > 0:
            ativadas.append((desvio, regra))

    if not ativadas:
        return None, None, None

    desvio, regra = max(ativadas, key=lambda par: par[0])
    return regra.nome, regra.correcao, float(desvio)
