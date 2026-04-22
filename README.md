# ComfyUI-Kanade-S2S

Nodos personalizados para ComfyUI que integran el [Kanade: A Simple Disentangled Tokenizer for Spoken Language Modeling](https://github.com/frothywater/kanade-tokenizer).

Estos nodos permiten codificar y decodificar voz utilizando el tokenizador Kanade. Incluye soporte para el último modelo `kanade-25hz-clean` usando el vocoder HiFT.

## Nodos Disponibles

* **Kanade Model Loader**: Descarga y carga los pesos del modelo Kanade y su Vocoder directamente en la carpeta estándar `models/kanade` de ComfyUI.
* **Kanade Encoder**: Recibe la entrada estándar `AUDIO` de ComfyUI, y extrae y separa los tokens de contenido (`KANADE_TOKENS`) y su embedding global (`KANADE_EMBEDDING`).
* **Kanade Decoder**: Toma los tokens y el embedding para reconstruirlos nuevamente a la salida estándar `AUDIO` de ComfyUI.

## Instalación

Para instalarlo, puedes hacerlo de dos formas:

1. Clonar este repositorio dentro de la carpeta `ComfyUI/custom_nodes`:
```bash
cd ComfyUI/custom_nodes
git clone <URL_DE_ESTE_REPO>
cd ComfyUI-Kanade-S2S
pip install -r requirements.txt
```
2. O bien, utilizar el ComfyUI Manager (si está registrado).

### Advertencia para usuarios de Windows / `flash-attn`

El modelo base de Kanade utiliza la librería `flash-attn` para lograr un cálculo eficiente de atención de ventana local.
Los usuarios de Windows a menudo se encuentran con problemas al intentar compilar esta biblioteca. Recomendamos instalarla utilizando el siguiente comando desde el entorno virtual de Python asociado a ComfyUI:

```bash
uv pip install flash-attn --no-build-isolation
```
*(Asegúrate de tener instalada la herramienta `ninja` en tu sistema, o la compilación será extremadamente lenta)*

Si la librería `flash-attn` no está disponible en tu entorno, el modelo realizará un fallback a la implementación convencional SDPA de PyTorch, lo que podría no garantizar la misma calidad de síntesis reportada en el paper original.
