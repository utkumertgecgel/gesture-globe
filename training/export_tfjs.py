"""
PyTorch → TensorFlow.js Model Dönüştürücü
==========================================
Eğitilmiş PyTorch modelini TF.js formatına dönüştürür.
Bu model web uygulamasında tarayıcıda çalışacak.

Kullanım:
  python export_tfjs.py
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn

# Model tanımları (train_models.py ile aynı)
class MLPModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)


def export_to_tfjs_via_onnx():
    """ONNX Runtime Web formatında export et (en kolay yol)"""
    
    MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
    WEB_MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "web-app", "models")
    os.makedirs(WEB_MODELS_DIR, exist_ok=True)
    
    # Label map'i kopyala
    label_map_src = os.path.join(MODELS_DIR, "label_map.json")
    if os.path.exists(label_map_src):
        import shutil
        shutil.copy(label_map_src, os.path.join(WEB_MODELS_DIR, "label_map.json"))
        print(f"Label map kopyalandi: {WEB_MODELS_DIR}/label_map.json")
    
    # ONNX modelini web klasorune kopyala
    onnx_src = os.path.join(MODELS_DIR, "mlp_best.onnx")
    if os.path.exists(onnx_src):
        import shutil
        shutil.copy(onnx_src, os.path.join(WEB_MODELS_DIR, "model.onnx"))
        print(f"ONNX model kopyalandi: {WEB_MODELS_DIR}/model.onnx")
    
    # Ayrica PyTorch'tan dogrudan weight'leri JSON olarak export et
    # (TF.js/ONNX Runtime olmadan da kullanilabilir basit format)
    export_weights_json(MODELS_DIR, WEB_MODELS_DIR)
    
    print("\nExport tamamlandi!")
    print(f"Dosyalar: {WEB_MODELS_DIR}")


def export_weights_json(models_dir, web_dir):
    """
    Model agirliklarini JSON formatinda export eder.
    Web app'te saf JavaScript ile inference yapilabilir.
    Bu yaklasim hicbir ek kutuphane gerektirmez.
    """
    
    # Model yukle
    input_size = 73  # 63 landmark + 10 mesafe
    num_classes = 5
    
    model = MLPModel(input_size, num_classes)
    model_path = os.path.join(models_dir, "mlp_best.pth")
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()
    
    # Tum katmanlarin weight ve bias'larini cikar
    weights = {}
    for name, param in model.named_parameters():
        weights[name] = param.detach().numpy().tolist()
    
    # BatchNorm running stats
    for name, buf in model.named_buffers():
        weights[name] = buf.detach().numpy().tolist()
    
    # Model mimarisini de kaydet
    model_info = {
        "type": "MLP",
        "input_size": input_size,
        "num_classes": num_classes,
        "layers": [
            {"type": "linear", "in": input_size, "out": 128},
            {"type": "batchnorm", "size": 128},
            {"type": "relu"},
            {"type": "linear", "in": 128, "out": 64},
            {"type": "batchnorm", "size": 64},
            {"type": "relu"},
            {"type": "linear", "in": 64, "out": 32},
            {"type": "relu"},
            {"type": "linear", "in": 32, "out": num_classes},
        ],
        "weights": weights
    }
    
    output_path = os.path.join(web_dir, "model.json")
    with open(output_path, 'w') as f:
        json.dump(model_info, f)
    
    file_size = os.path.getsize(output_path) / 1024
    print(f"Model JSON export: {output_path} ({file_size:.1f} KB)")
    
    # Test: inference kontrol
    test_input = np.random.randn(input_size).astype(np.float32)
    with torch.no_grad():
        tensor_input = torch.FloatTensor(test_input).unsqueeze(0)
        output = model(tensor_input)
        probs = torch.softmax(output, dim=1).numpy()[0]
    
    print(f"Test inference basarili. Ornek cikti: {probs}")


if __name__ == "__main__":
    export_to_tfjs_via_onnx()
