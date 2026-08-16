import base64
import io

import cv2
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
import streamlit as st
from PIL import Image

# NOTE: On Windows you may need to set the tesseract executable path, e.g.:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Enable OpenCL (if available)
cv2.ocl.setUseOpenCL(True)


# ---------- Geometry & Contour Utilities ----------
def order_points(pts: np.ndarray) -> list[list[int]]:
    """Order 4 corner points as top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect.astype("int").tolist()


def find_dest(pts: list[list[int]]) -> list[list[int]]:
    """Compute destination rectangle for perspective transform."""
    (tl, tr, br, bl) = pts
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    destination_corners = np.array(
        [[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]],
        dtype="float32",
    )
    return order_points(destination_corners)


def fast_dilate(img: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(img, kernel, iterations=1)


def canny_edges(gray: np.ndarray) -> np.ndarray:
    """Dynamic-threshold Canny edge detection (as in [1], [7], [8] of your proposal)."""
    v = np.median(gray)
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))
    edges = cv2.Canny(gray, lower, upper)
    return edges


# ---------- Core Document Scan Pipeline ----------
def scan(img: np.ndarray) -> np.ndarray:
    """
    Detect the largest quadrilateral contour, apply perspective transform,
    and return a rectified, scanned-like BGR image.
    """
    dim_limit = 1080
    h, w = img.shape[:2]
    if max(h, w) > dim_limit:
        scale = dim_limit / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale)

    orig_img = img.copy()

    # Preprocessing: grayscale + Gaussian blur for noise reduction
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection and dilation to strengthen document borders
    edges = canny_edges(gray_blur)
    edges = fast_dilate(edges, kernel_size=5)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    img_area = h * w
    min_doc_ratio = 0.15  # document must be at least 15% of image (avoid tiny regions)
    page = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    if not page:
        return orig_img

    corners = None
    for c in page:
        if cv2.contourArea(c) < min_doc_ratio * img_area:
            continue  # skip small contours (text blocks, headers)
        epsilon = 0.02 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        if len(approx) == 4:
            corners = approx
            break

    if corners is None:
        return orig_img  # no suitable document contour: show full image

    corners = np.array(corners, dtype="float32").reshape(4, 2)
    corners = order_points(corners)
    dst = find_dest(corners)

    M = cv2.getPerspectiveTransform(np.float32(corners), np.float32(dst))
    final = cv2.warpPerspective(
        orig_img, M, (dst[2][0], dst[2][1]), flags=cv2.INTER_LINEAR
    )
    return final


# ---------- OCR Utilities ----------
def preprocess_for_ocr(scanned_bgr: np.ndarray) -> np.ndarray:
    """
    Convert rectified BGR image to high-contrast binary image suitable for OCR.
    Uses grayscale, adaptive thresholding, and light morphology.
    """
    gray = cv2.cvtColor(scanned_bgr, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold (Gaussian) to handle uneven lighting [2], [8], [9].
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        25,
        10,
    )

    # Optional small dilation/erosion to clean noise
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return binary


def run_ocr(scanned_bgr: np.ndarray) -> str:
    """
    Run Tesseract OCR on the rectified document using a suitable PSM.
    PSM 6 assumes a uniform block of text, matching your methodology section.
    """
    preprocessed = preprocess_for_ocr(scanned_bgr)
    config = "--oem 3 --psm 6"
    text = pytesseract.image_to_string(preprocessed, config=config)
    return text


def get_pdf_download_link(images: list[np.ndarray], filename: str = "scanned_document.pdf") -> str:
    pdf_bytes = io.BytesIO()
    pil_images = [Image.fromarray(img[:, :, ::-1]).convert("RGB") for img in images]
    pil_images[0].save(pdf_bytes, format="PDF", save_all=True, append_images=pil_images[1:])
    pdf_bytes.seek(0)
    b64 = base64.b64encode(pdf_bytes.read()).decode()
    return f'<a class="download-btn" href="data:file/pdf;base64,{b64}" download="{filename}">Download All as PDF</a>'


def get_text_download_link(text: str, filename: str = "extracted_text.txt") -> str:
    text_bytes = text.encode("utf-8")
    b64 = base64.b64encode(text_bytes).decode()
    return f'<a class="download-btn" href="data:text/plain;base64,{b64}" download="{filename}">Download Extracted Text</a>'


# ---------- Streamlit App ----------
st.set_page_config(
    page_title="Document Scanner | OCR",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a cleaner, modern look
st.markdown("""
<style>
    /* Main container & typography */
    .main .block-container { padding: 2rem 3rem; max-width: 1400px; }
    h1, h2, h3 { font-weight: 600 !important; }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1d29 0%, #0f1117 100%);
    }
    [data-testid="stSidebar"] .stMarkdown { color: #e2e8f0 !important; }
    
    /* Document card */
    .doc-card {
        background: linear-gradient(145deg, #1e2130 0%, #161922 100%);
        border: 1px solid rgba(99, 179, 237, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.3);
    }
    .doc-card h3 {
        color: #63b3ed !important;
        margin-bottom: 1rem !important;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Image container with subtle frame */
    .img-frame {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        border: 1px solid rgba(255,255,255,0.06);
    }
    
    /* OCR text section */
    .ocr-section {
        background: rgba(15, 17, 23, 0.8);
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #63b3ed;
        margin-top: 1rem;
    }
    
    /* Download buttons in sidebar */
    .download-btn {
        display: inline-block;
        background: linear-gradient(135deg, #3182ce 0%, #2c5282 100%);
        color: white !important;
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        text-decoration: none !important;
        font-weight: 500;
        margin: 0.3rem 0;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .download-btn:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(49, 130, 206, 0.4);
    }
    
    /* Empty state / hero */
    .hero {
        text-align: center;
        padding: 4rem 2rem;
    }
    .hero h1 { font-size: 2.2rem !important; margin-bottom: 0.5rem !important; }
    .hero p { color: #94a3b8; font-size: 1.1rem; }
    
    /* Expander styling */
    .streamlit-expanderHeader { font-weight: 500 !important; }
</style>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 📄 Document Scanner")
    st.markdown("*Scan & extract text with OCR*")
    st.markdown("---")
    
    uploaded_files = st.file_uploader(
        "**Upload document images**",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help="PNG, JPG, JPEG up to 200MB each",
    )

if not uploaded_files:
    st.markdown('<div class="hero">', unsafe_allow_html=True)
    st.title("📄 Document Scanner with OCR")
    st.markdown("Upload document images in the sidebar to scan and extract text automatically.")
    st.markdown("Supports multiple files — each will be rectified and processed for OCR.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# ---------- Process uploaded files ----------
scanned_images: list[np.ndarray] = []
ocr_texts: list[str] = []

st.markdown("### Scanned documents & extracted text")

for idx, uploaded_file in enumerate(uploaded_files, start=1):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    scanned = scan(image)
    scanned_images.append(scanned)

    text = run_ocr(scanned)
    ocr_texts.append(text)

    st.markdown(f"#### 📑 Document {idx}")
    
    col_img, col_txt = st.columns([3, 2])
    
    with col_img:
        st.image(scanned, channels="BGR", width=600)
    
    with col_txt:
        with st.expander("**Show extracted text**"):
            st.text_area("Extracted text", value=text, height=280, key=f"ocr_text_{idx}")
    
    st.markdown("---")

# ---------- Sidebar downloads ----------
st.sidebar.markdown("---")
st.sidebar.markdown("**Downloads**")

st.sidebar.markdown(
    f'<div style="margin-top:0.5rem;">{get_pdf_download_link(scanned_images)}</div>',
    unsafe_allow_html=True,
)

combined_text = "\n\n".join(
    [f"--- Document {i + 1} ---\n{t}" for i, t in enumerate(ocr_texts)]
)
st.sidebar.markdown(
    f'<div style="margin-top:0.5rem;">{get_text_download_link(combined_text)}</div>',
    unsafe_allow_html=True,
)
