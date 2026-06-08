"""
Extração de keypoints via MediaPipe Pose Landmarker (Tasks API).
Parâmetros e política de descarte idênticos aos do treinamento.
"""

import time
import cv2
import numpy as np
import mediapipe as mp


BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MIN_SCORE = 0.5


class ExtratorKeypoints:
    """
    Envolve um PoseLandmarker em modo VIDEO. Timestamps monótonos derivados
    de time.monotonic_ns garantem que frames de webcam sejam aceitos.
    """

    def __init__(self, caminho_modelo_task: str, min_confianca: float = MIN_SCORE):
        self._options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=caminho_modelo_task),
            running_mode=VisionRunningMode.VIDEO,
            min_pose_detection_confidence=min_confianca,
            min_tracking_confidence=min_confianca,
            num_poses=1,
        )
        self._landmarker = PoseLandmarker.create_from_options(self._options)
        self._t0_ns = time.monotonic_ns()
        self._ultimo_ts_ms = -1

    def detectar(self, frame_bgr):
        """
        Processa um frame BGR da webcam.

        Retorna tupla (landmarks_desenho, coords_buffer):
          landmarks_desenho: lista de 33 objetos com .x/.y/.visibility para
                             desenhar o esqueleto; None se nenhuma pose detectada.
          coords_buffer:     ndarray (99,) pronto para entrar no buffer, ou None
                             se a visibilidade média foi < 0.5 (frame descartado).
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        ts_ms = (time.monotonic_ns() - self._t0_ns) // 1_000_000
        if ts_ms <= self._ultimo_ts_ms:
            ts_ms = self._ultimo_ts_ms + 1
        self._ultimo_ts_ms = ts_ms

        resultado = self._landmarker.detect_for_video(mp_image, int(ts_ms))

        if not resultado.pose_landmarks:
            return None, None

        kps = resultado.pose_landmarks[0]
        coords = np.array(
            [[kp.x, kp.y, kp.visibility] for kp in kps],
            dtype=np.float64,
        ).flatten()

        visibilidade_media = float(np.mean([kp.visibility for kp in kps]))
        if visibilidade_media < MIN_SCORE:
            return kps, None

        return kps, coords

    def fechar(self):
        try:
            self._landmarker.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.fechar()
