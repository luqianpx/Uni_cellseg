import numpy as np, cv2
from skimage import color
from dataclasses import dataclass

@dataclass
class ReinhardNormalizer:
    """Simple Reinhard color normalization between LAB statistics of image and reference tile."""
    ref_mean: np.ndarray
    ref_std: np.ndarray

    @staticmethod
    def from_reference_lab(ref_lab_img: np.ndarray):
        lab = ref_lab_img.astype(np.float32)
        mean = lab.reshape(-1, 3).mean(axis=0)
        std = lab.reshape(-1, 3).std(axis=0) + 1e-6
        return ReinhardNormalizer(mean, std)

    def __call__(self, img_rgb: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        mean = lab.reshape(-1,3).mean(axis=0)
        std = lab.reshape(-1,3).std(axis=0) + 1e-6
        lab_norm = (lab - mean) / std * self.ref_std + self.ref_mean
        lab_norm = np.clip(lab_norm, 0, 255).astype(np.uint8)
        return cv2.cvtColor(lab_norm, cv2.COLOR_LAB2RGB)

# Minimal Macenko implementation (approximate for general stains)
def macenko_normalize(img_rgb: np.ndarray, Io=240, alpha=1, beta=0.15):
    # Convert to optical density OD
    img = img_rgb.astype(np.uint8)
    img = np.clip(img, 1, 255)
    OD = -np.log((img.astype(np.float32))/Io)
    ODhat = OD[~np.any(img==0, axis=2)]
    # SVD on OD
    _, V = np.linalg.eig(np.cov(ODhat.T))
    V = np.real(V)
    # Choose the two largest eigenvectors
    if V.shape[1] >= 2:
        v1, v2 = V[:, :2].T
    else:
        v1, v2 = V[:, 0], V[:, 0]
    # Project on plane spanned by v1 and v2
    That = OD.reshape((-1,3)).dot(np.vstack((v1, v2)).T)
    phi = np.arctan2(That[:,1], That[:,0])
    minPhi = np.percentile(phi, beta*100)
    maxPhi = np.percentile(phi, (1-beta)*100)
    vMin = np.array([np.cos(minPhi), np.sin(minPhi)])
    vMax = np.array([np.cos(maxPhi), np.sin(maxPhi)])
    HE = np.vstack((vMin, vMax)).T
    # Pseudoinverse to obtain concentrations
    C = np.linalg.lstsq(HE, That.T, rcond=None)[0]
    # Normalize concentrations
    maxC = np.percentile(C, 99, axis=1)
    C2 = C / (maxC[:,None] + 1e-6)
    # Reconstruct
    # Reference stain matrix for H&E-like; for general stains this is approximate
    # Columns are stain OD vectors
    M = np.array([[0.65, 0.70],
                  [0.70, 0.99],
                  [0.29, 0.11]], dtype=np.float32)
    OD_norm = (M.dot(C2)).T.reshape(OD.shape)
    Io_arr = np.array(Io).reshape((1,1,1))
    img_norm = (Io_arr * np.exp(-OD_norm)).astype(np.uint8)
    img_norm = np.clip(img_norm, 0, 255)
    return img_norm

def to_uint8(img):
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = np.clip(img, 0, 1) * 255.0
        img = img.astype(np.uint8)
    return img
