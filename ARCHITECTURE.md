# Arquitectura del Sistema de Reconocimiento Facial (FaceID Backend)

## 1. Flujo de Procesamiento de Imágenes
El sistema sigue un pipeline secuencial optimizado para baja latencia:

1.  **Recepción**: La imagen llega como bytes (UploadFile) al endpoint REST.
2.  **Decodificación**: OpenCV decodifica los bytes a una matriz BGR (numpy array).
3.  **Detección (Detection)**: El modelo RetinaFace (incluido en InsightFace) escanea la imagen buscando rostros.
4.  **Validación de Calidad (Liveness/Quality)**:
    *   Se verifica el "score" de detección.
    *   Se evalúa la pose (pitch, yaw, roll) para rechazar ángulos extremos.
    *   *Nota*: La detección de parpadeo real requiere video stream. Para una imagen estática, analizamos la calidad textural y la coherencia 3D inferida.
5.  **Alineación (Alignment)**: Se realiza una transformación afín (Affine Transformation) usando 5 puntos clave (landmarks) para normalizar la rotación y escala del rostro.
6.  **Extracción (Embedding)**: El modelo ArcFace (ResNet100/50) procesa el rostro alineado (112x112 px) y emite un vector de 512 dimensiones (float32).
7.  **Indexación/Búsqueda (Matching)**:
    *   En `/enroll`: El vector se guarda junto con el ID del usuario en SQLite.
    *   En `/verify`: El vector se compara contra la base de datos usando Similitud Coseno (Cosine Similarity).

## 2. Componentes Principales

*   **API Gateway (FastAPI)**: Maneja las peticiones HTTP, validación de tipos y manejo de errores.
*   **Vision Engine (InsightFace/ONNX)**:
    *   *Detector*: RetinaFace (mnet0.25 o buffalo_s para rapidez).
    *   *Recognizer*: ArcFace (buffalo_l para precisión).
*   **Storage (SQLite)**: Almacena metadatos de usuario y los embeddings serializados.
*   **Matcher (Vector Search)**: Lógica de búsqueda lineal (para <1000 usuarios es más rápido que índices complejos como FAISS) usando Producto Punto/Coseno.

## 3. Seguridad y Privacidad

*   **Liveness**: Se rechazan rostros con baja probabilidad de ser reales (basado en thresholds de detección).
*   **Privacidad**: Las imágenes se descartan inmediatamente después del procesamiento (RAM release). Solo se persisten los vectores matemáticos irreversibles.
*   **Anti-Replay**: Recomendamos que el cliente envíe un timestamp firmado o nonce en los headers (implementación básica incluida en headers).
