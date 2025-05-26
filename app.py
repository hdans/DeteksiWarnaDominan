import streamlit as st
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
from PIL import Image
import matplotlib.pyplot as plt
from kneed import KneeLocator


def get_dominant_colors(image, num_colors=5):
    """
    Ekstrak warna dominan dari gambar menggunakan KMeans clustering.

    Args:
        image (PIL.Image.Image): Objek gambar PIL.
        num_colors (int): Jumlah warna dominan yang ingin diekstrak.

    Returns:
        numpy.ndarray: Array NumPy berisi nilai RGB dari warna dominan.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    img_np = np.array(image)
    pixels = img_np.reshape(-1, 3)
    pixels = np.float32(pixels)

    kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
    kmeans.fit(pixels)

    colors = kmeans.cluster_centers_
    labels = kmeans.predict(pixels)
    label_counts = Counter(labels)
    sorted_colors = [colors[i] for i, _ in label_counts.most_common()]

    return np.array(sorted_colors, dtype=int)

def rgb_to_hex(rgb):
    """
    Konversi nilai RGB ke format heksadesimal.

    Args:
        rgb (tuple/list/numpy.ndarray): Tuple atau array berisi nilai RGB (0-255).

    Returns:
        str: String kode heksadesimal warna.
    """
    return f"#{int(rgb[0]):02x}{int(rgb[1]):02x}{int(rgb[2]):02x}"

def calculate_wcss(pixels, max_k=15):
    """
    Menghitung Within-Cluster Sum of Squares (WCSS) untuk berbagai nilai k.

    Args:
        pixels (numpy.ndarray): Array piksel gambar (Nx3).
        max_k (int): Nilai k maksimum yang akan diuji.

    Returns:
        list: Daftar nilai WCSS untuk setiap k dari 1 hingga max_k.
    """
    wcss = []
    num_samples = min(len(pixels), 100000)
    sample_indices = np.random.choice(len(pixels), num_samples, replace=False)
    sampled_pixels = pixels[sample_indices]

    for i in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
        kmeans.fit(sampled_pixels)
        wcss.append(kmeans.inertia_)
    return wcss

def find_optimal_k_elbow(wcss_values, k_range):
    """
    Mendeteksi titik 'siku' pada kurva WCSS untuk menentukan k optimal.

    Args:
        wcss_values (list): Daftar nilai WCSS.
        k_range (list): Daftar nilai k yang sesuai dengan WCSS.

    Returns:
        int: Nilai k optimal yang terdeteksi.
    """
    try:
        kneedle = KneeLocator(k_range, wcss_values, S=1.0, curve='convex', direction='decreasing', interp_method='polynomial')
        # Pastikan k minimal 2 jika elbow terdeteksi 1, karena untuk palet warna butuh minimal 2
        return max(2, kneedle.elbow) if kneedle.elbow is not None else 5
    except Exception as e:
        st.warning(f"Gagal mendeteksi titik elbow, menggunakan k=5 sebagai default. Error: {e}")
        return 5

st.set_page_config(
    page_title="Menentukan Warna Dominan",
    layout="wide", 
    initial_sidebar_state="auto"
)

st.markdown(
    """
    <style>
    body {
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        color: #E0E0E0; /* Light text for dark background */
        background-color: #1a1a1a; /* Very dark background */
    }
    .stApp {
        background-color: #212121; /* Slightly lighter dark background for content */
        box-shadow: 0 4px 15px rgba(0,0,0,0.5); /* More pronounced shadow */
        border-radius: 12px;
        padding: 2.5rem; /* Increased padding */
        margin: 2rem auto;
        border: 1px solid #333; /* Subtle border for definition */
    }

    h1, h2, h3, h4, h5, h6 {
        color: #F8F8F8; /* Bright text for headers */
        font-weight: 700; /* Bolder headers */
    }
    h1 {
        font-size: 2.8em; /* Larger title */
        text-align: center;
        margin-bottom: 1em;
        text-shadow: 0 0 10px rgba(255,255,255,0.1); /* Soft text glow */
    }
    h2 {
        font-size: 2em;
        margin-top: 2em; /* More spacing */
        border-bottom: 1px solid #444; /* Darker subtle separator */
        padding-bottom: 0.8em;
    }

    .stFileUploader label {
        font-size: 1.1em;
        font-weight: 500;
        color: #61DAFB; /* Cyan accent for uploader */
    }
    .stFileUploader > div > button {
        background-color: #61DAFB; /* Cyan button */
        color: #212121; /* Dark text on bright button */
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 1em;
        transition: background-color 0.3s, transform 0.2s;
    }
    .stFileUploader > div > button:hover {
        background-color: #21A1F1; /* Slightly darker cyan */
        transform: translateY(-3px); /* More pronounced lift */
    }

    .stButton > button {
        background-color: #8A2BE2; /* Violet accent color */
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-size: 1em;
        cursor: pointer;
        transition: background-color 0.3s, transform 0.2s;
    }
    .stButton > button:hover {
        background-color: #6A1FB8; /* Darker violet */
        transform: translateY(-2px);
    }

    .color-item-wrapper {
        /* This wrapper defines the size of each grid cell item */
        width: calc(33.33% - 20px) !important; /* Forces 3 items per row, subtracting gap */
        display: flex !important; /* Use flex inside to stack content vertically */
        flex-direction: column !important;
        align-items: center !important; /* Center horizontally */
        justify-content: flex-start !important; /* Align content to top of item wrapper */
        box-sizing: border-box !important; /* Crucial for width calculation */
        margin-bottom: 20px !important; /* Space between rows */
        text-align: center !important;
    }

    .color-box {
        width: 100px !important; /* Fixed width for the color square */
        height: 100px !important; /* Fixed height for the color square */
        display: block !important;
        margin: 0 auto 10px auto !important; /* Center within wrapper and space below */
        border-radius: 12px !important;
        border: 2px solid rgba(255,255,255,0.1) !important;
        cursor: pointer !important;
        transition: transform 0.2s ease-in-out, box-shadow 0.2s, border-color 0.2s !important;
    }
    .color-box:hover {
        transform: scale(1.05) !important;
        box-shadow: 0 8px 25px rgba(0,255,255,0.3) !important;
        border-color: #61DAFB !important;
    }
    .hex-code {
        font-family: 'Fira Code', 'Cascadia Code', monospace !important;
        font-size: 1em !important;
        margin-top: 8px !important;
        color: #BBBBBB !important;
        text-align: center !important;
        background-color: #333333 !important;
        padding: 8px !important;
        border-radius: 8px !important;
        cursor: pointer !important;
        transition: background-color 0.2s, color 0.2s !important;
        user-select: all !important;
        max-width: 150px !important; /* Max width for hex code */
        word-break: break-all !important;
    }
    .hex-code:hover {
        background-color: #444444 !important;
        color: #FFFFFF !important;
    }

    .stAlert {
        border-radius: 8px !important;
        padding: 1.2em !important;
        margin-top: 1.5em !important;
        font-size: 1em !important;
        border: none !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3) !important;
    }
    .stAlert.info {
        background-color: #31475A !important;
        color: #9ECFFB !important;
    }
    .stAlert.success {
        background-color: #3A5F3A !important;
        color: #BEEBBE !important;
    }
    .stAlert.warning {
        background-color: #6A5C33 !important;
        color: #FDE8A5 !important;
    }

    .stPlotlyChart, .stImage {
        border-radius: 12px !important;
        overflow: hidden !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3) !important;
        border: 1px solid #444 !important;
    }

    hr {
        border: 0 !important;
        height: 1px !important;
        background: linear-gradient(to right, rgba(255,255,255,0), #555, rgba(255,255,255,0)) !important;
        margin: 3em 0 !important;
    }

    p {
        line-height: 1.7 !important;
        margin-bottom: 1.2em !important;
        color: #D0D0D0 !important;
    }
    a {
        color: #61DAFB !important;
        text-decoration: none !important;
    }
    a:hover {
        text-decoration: underline !important;
    }

    .color-palette-container {
        display: flex !important; /* Use Flexbox */
        flex-wrap: wrap !important; /* Allow items to wrap to next line */
        justify-content: center !important; /* Center items if they don't fill the last row */
        gap: 20px !important; /* Space between items */
        padding: 15px !important;
        background-color: #2B2B2B !important;
        border-radius: 12px !important;
        margin-top: 1.5em !important;
        box-shadow: inset 0 0 10px rgba(0,0,0,0.3) !important;
        /* Tambahkan ini jika container Streamlit membatasi lebar */
        max-width: 100% !important;
    }

    @media (max-width: 768px) {
        .stApp {
            padding: 1.5rem !important;
            margin: 1rem auto !important;
            border-radius: 8px !important;
        }
        h1 {
            font-size: 2em !important;
        }
        h2 {
            font-size: 1.5em !important;
        }
        .color-item-wrapper {
            flex: 0 0 calc(50% - 15px) !important; /* 2 items per row on small screens */
            max-width: calc(50% - 15px) !important;
        }
        .color-palette-container {
            gap: 15px !important; /* Smaller gap on small screens */
        }
        .color-box {
            width: 80px !important; /* Even smaller for mobile */
            height: 80px !important;
        }
        .hex-code {
            max-width: 80px !important;
            font-size: 0.8em !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("140810230058 - Danish Rahadian Mirza Effendi")
st.markdown(
    """
    Selamat datang di program **Menentukan Warna Dominan Gambar**. Platform ini dibuat untuk
    mengekstrak representasi visual warna dominan dari gambar yang diunggah. Dengan mengaplikasikan
    algoritma _K-Means Clustering_ yang efisien, dilengkapi dengan validasi melalui **Metode _Elbow_**,
    sistem ini mampu secara otomatis mengidentifikasi dan menyajikan spektrum warna kunci yang merefleksikan
    karakteristik visual gambar.
    """
)

uploaded_file = st.file_uploader("Unggah Gambar (JPG, JPEG, PNG, WEBP)", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file) 

    col_img, col_info = st.columns([2, 1])

    with col_img:
        st.image(image, caption='Gambar yang Diunggah', use_container_width=True)

    with col_info:
        st.markdown("### Detail Gambar")
        st.write(f"**Nama Berkas:** `{uploaded_file.name}`")
        st.write(f"**Ukuran:** `{image.width} x {image.height} piksel`")
        st.write(f"**Format:** `{image.format}`")
        st.markdown(
            """
            <br>
            <small style="color: #BBBBBB;">
            Gambar ini akan diproses untuk ekstraksi warna dominan.
            </small>
            """, unsafe_allow_html=True
        )

    img_np_for_processing = np.array(image.convert("RGB"))
    pixels_for_processing = np.float32(img_np_for_processing.reshape(-1, 3))

    st.subheader("Analisis Klustering Optimal (Metode Elbow)")
    st.markdown(
        """
        Pada segmen ini, sistem melakukan analisis **_Within-Cluster Sum of Squares_ (WCSS)** untuk berbagai
        konfigurasi kluster. Grafik di bawah memvisualisasikan hubungan antara nilai WCSS dan jumlah
        kluster ($k$), yang krusial dalam identifikasi titik 'siku'. Titik ini secara heuristik
        menunjukkan jumlah kluster yang paling merepresentasikan struktur data warna dengan optimal.
        """
    )

    with st.spinner("Melakukan analisis WCSS untuk penentuan $k$ optimal..."):
        max_k_to_test = 10
        k_range = list(range(1, max_k_to_test + 1))
        wcss_values = calculate_wcss(pixels_for_processing, max_k=max_k_to_test)
    
        optimal_k = find_optimal_k_elbow(wcss_values, k_range)

        fig, ax = plt.subplots(figsize=(10, 6)) 
        ax.plot(k_range, wcss_values, marker='o', linestyle='-', color='#61DAFB', linewidth=2.5) 
        ax.axvline(x=optimal_k, color='#FF5733', linestyle='--', label=f'k Optimal = {optimal_k}', linewidth=2) 
        ax.set_title('Kurva Elbow untuk Penentuan Kluster Optimal', fontsize=18, color='#F8F8F8')
        ax.set_xlabel('Jumlah Kluster (k)', fontsize=14, color='#E0E0E0')
        ax.set_ylabel('WCSS (Within-Cluster Sum of Squares)', fontsize=14, color='#E0E0E0')
        ax.tick_params(axis='both', which='major', colors='#BBBBBB')
        ax.set_facecolor('#2B2B2B') 
        fig.patch.set_facecolor('#2B2B2B') 
        ax.legend(fontsize=12, facecolor='#2B2B2B', edgecolor='#444', labelcolor='#F8F8F8')
        ax.grid(True, linestyle=':', alpha=0.5, color='#555')
        plt.tight_layout()
        st.pyplot(fig)

        st.success(f"Berdasarkan analisis  menggunakan Metode Elbow, jumlah kluster yang terdeteksi adalah: **{optimal_k}**.")

    st.subheader("Visualisasi Palet Warna Dominan")
    st.markdown(
        f"""
        Berikut disajikan **{optimal_k} warna dominan** yang diekstraksi dari gambar yang diunggah.
        Setiap representasi warna disertai dengan kode heksadesimalnya, yang dapat disalin.
        """
    )

    with st.spinner(f"Mengekstrak {optimal_k} warna dominan..."):
        dominant_colors = get_dominant_colors(image, num_colors=optimal_k)

        st.markdown('<div class="color-palette-container">', unsafe_allow_html=True)
        for i, color_rgb in enumerate(dominant_colors):
            hex_code = rgb_to_hex(color_rgb)
            st.markdown(
                f"""
                <div class="color-item-wrapper">
                    <div class="color-box" style="background-color: {hex_code};"></div>
                    <div class="hex-code" onclick="navigator.clipboard.writeText('{hex_code}'); alert('Kode heksadesimal disalin: {hex_code}');">
                        {hex_code}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)

        st.info("Klik pada kode heksadesimal untuk menyalinnya ke _clipboard_.")

st.subheader("Metodologi Eksplorasi Warna")
st.markdown(
    """
    Implementasi ini memanfaatkan **_K-Means Clustering_**, sebuah algoritma pembelajaran tanpa label,
    untuk mengelompokkan piksel gambar berdasarkan kesamaan nilai dalam ruang warna RGB.
    Prinsip dasar melibatkan representasi setiap piksel sebagai titik data tiga dimensi (R, G, B),
    yang kemudian dikelompokkan menjadi $k$ kluster. _Centroid_ dari setiap kluster ini berfungsi sebagai
    representasi dari warna dominan yang ditemukan dalam gambar.

    Penentuan jumlah kluster ($k$) yang paling representatif difasilitasi oleh **Metode _Elbow_**.
    Pendekatan ini mengevaluasi **_Within-Cluster Sum of Squares (WCSS)_**, yaitu agregat kuadrat
    jarak antara setiap titik data dan _centroid_ klusternya. Visualisasi WCSS terhadap $k$
    menghasilkan sebuah kurva yang secara karakteristik menampilkan titik _elbow_. Titik ini
    mengindikasikan di mana penambahan jumlah kluster tidak lagi memberikan pengurangan WCSS yang
    substansial, menandai $k$ optimal yang menyeimbangkan kohesi kluster dengan kompleksitas model.
    Deteksi titik _elbow_ ini diotomatisasi untuk meningkatkan efisiensi proses.
    """
)
st.markdown(
    """
    ---
    140810230058 - Danish Rahadian Mirza Effendi
    """
)