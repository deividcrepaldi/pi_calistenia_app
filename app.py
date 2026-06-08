"""
Ponto de entrada da aplicação.

Uso:
    python app.py

Ao iniciar:
  - silencia logs verbosos do TensorFlow
  - instancia a janela principal
  - entra no mainloop do Tkinter

A janela gerencia internamente as duas telas:
  Tela 1: seleção de exercício
  Tela 2: análise em tempo real (webcam + inferência LSTM)
"""

import os
import sys


def _configurar_ambiente():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("GLOG_minloglevel", "2")


def main():
    _configurar_ambiente()
    # Import tardio: só depois de silenciar logs do TF.
    from interface.aplicacao import Aplicacao

    app = Aplicacao()
    try:
        app.mainloop()
    except KeyboardInterrupt:
        app.destroy()
        sys.exit(0)


if __name__ == "__main__":
    main()
