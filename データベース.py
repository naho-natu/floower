import os
import shutil
from datetime import datetime
import torch
from torch import nn, optim
from torchvision import models, transforms, datasets
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
import wikipedia
import re

# === 設定 ===
ROOT_DIR = "./data"
MODEL_PATH = "flowers102_resnet18.pth"
ZUKAN_DIR = "./zukan"
os.makedirs(ZUKAN_DIR, exist_ok=True)

# === デバイス設定 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 画像変換 ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# === データセット準備 ===
train_ds = datasets.Flowers102(root=ROOT_DIR, split="train", download=False, transform=transform)
val_ds   = datasets.Flowers102(root=ROOT_DIR, split="val",   download=False, transform=transform)
class_names = train_ds.classes
NUM_CLASSES = len(class_names)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=32)

# === 英語→日本語 名辞書 ===
# === 英語名→日本語名対応辞書 ===
en2jp = {
    "pink primrose": "ピンクのサクラソウ",
    "hard-leaved pocket orchid": "カチカチポケットラン",
    "canterbury bells": "カンタベリー・ベル",
    "sweet pea": "スイートピー",
    "english marigold": "イングリッシュマリーゴールド",
    "tiger lily": "オニユリ",
    "moon orchid": "ファレノプシス（胡蝶蘭）",
    "bird of paradise": "ゴクラクチョウカ（極楽鳥花）",
    "monkshood": "トリカブト",
    "globe thistle": "ヒゴタイ",
    "snapdragon": "キンギョソウ",
    "colt’s foot": "フキタンポポ",
    "king protea": "キングプロテア",
    "spear thistle": "アザミ",
    "yellow iris": "キショウブ",
    "globe-flower": "セイヨウキンバイ",
    "purple coneflower": "ムラサキバレンギク",
    "peruvian lily": "アルストロメリア",
    "balloon flower": "キキョウ",
    "giant white arum lily": "ジャイアントカラー",
    "fire lily": "ヒメヒオウギズイセン",
    "pincushion flower": "マツムシソウ",
    "fritillary": "バイモ",
    "red ginger": "ショウガの花（レッドジンジャー）",
    "grape hyacinth": "ムスカリ",
    "corn poppy": "ヒナゲシ",
    "prince of wales feathers": "センニチコウ",
    "stemless gentian": "フデリンドウ",
    "artichoke": "アーティチョーク（チョウセンアザミ）",
    "sweet william": "ナデシコ",
    "carnation": "カーネーション",
    "garden phlox": "フロックス",
    "love in the mist": "ニゲラ",
    "mexican aster": "コスモス",
    "alpine sea holly": "エリンジウム",
    "ruby-lipped cattleya": "カトレア",
    "cape flower": "ケープフラワー",
    "great masterwort": "マスターワート",
    "siam tulip": "クルクマ（シャムチューリップ）",
    "lenten rose": "クリスマスローズ",
    "barbeton daisy": "ガーベラ",
    "daffodil": "スイセン",
    "sword lily": "グラジオラス",
    "poinsettia": "ポインセチア",
    "bolero deep blue": "ボレロ・ディープブルー（バラの一種）",
    "wallflower": "ウォールフラワー（ニオイアラセイトウ）",
    "marigold": "マリーゴールド",
    "buttercup": "キンポウゲ",
    "oxeye daisy": "フランスギク",
    "common dandelion": "セイヨウタンポポ",
    "petunia": "ペチュニア",
    "wild pansy": "野生スミレ（パンジー）",
    "primula": "サクラソウ属",
    "sunflower": "ヒマワリ",
    "pelargonium": "ペラルゴニウム（ゼラニウムの一種）",
    "bishop of llandaff": "ビショップ・オブ・ランダフ（ダリアの一種）",
    "gaura": "ガウラ",
    "geranium": "ゼラニウム",
    "orange dahlia": "オレンジのダリア",
    "pink-yellow dahlia": "ピンクと黄色のダリア",
    "cautleya spicata": "カウレヤ・スピカタ",
    "japanese anemone": "シュウメイギク（秋明菊）",
    "black-eyed susan": "ルドベキア",
    "silverbush": "シルバーブッシュ",
    "californian poppy": "カリフォルニアポピー",
    "osteospermum": "アフリカンデージー",
    "spring crocus": "クロッカス",
    "bearded iris": "ドイツアヤメ",
    "windflower": "アネモネ",
    "tree poppy": "ツリーポピー",
    "gazania": "ガザニア",
    "azalea": "ツツジ",
    "water lily": "スイレン",
    "rose": "バラ",
    "thorn apple": "チョウセンアサガオ",
    "morning glory": "アサガオ",
    "passion flower": "トケイソウ",
    "lotus": "ハス",
    "toad lily": "ホトトギス",
    "anthurium": "アンスリウム",
    "frangipani": "プルメリア",
    "clematis": "クレマチス",
    "hibiscus": "ハイビスカス",
    "columbine": "オダマキ",
    "desert-rose": "アデニウム（砂漠のバラ）",
    "tree mallow": "モクフヨウ",
    "magnolia": "モクレン",
    "cyclamen": "シクラメン",
    "watercress": "クレソン（花ではないが含まれている）",
    "canna lily": "カンナ",
    "hippeastrum": "ヒッペアストルム（アマリリス）",
    "bee balm": "モナルダ（ベルガモット）",
    "ball moss": "ボールモス（ティランジア）",
    "foxglove": "ジギタリス",
    "bougainvillea": "ブーゲンビリア",
    "camellia": "ツバキ",
    "mallow": "アオイ",
    "mexican petunia": "メキシカンペチュニア",
    "bromelia": "ブロメリア属",
    "blanket flower": "ガイラルディア",
    "trumpet creeper": "ノウゼンカズラ",
    "blackberry lily": "ヒオウギ"
}

# === モデル準備 ===
model = models.resnet18(pretrained=True)
for p in model.parameters(): p.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(device)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
else:
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
    for epoch in range(3):
        tot = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            tot += loss.item()
        print(f"Epoch {epoch+1} loss: {tot/len(train_loader):.4f}")
    torch.save(model.state_dict(), MODEL_PATH)

model.eval()

wikipedia.set_lang("en")

def simple_translate_habitat(text):
    repl = {
        r"native to": "に自生し",
        r"found in":   "で見られ",
        r"endemic to": "に固有で",
        r"distributed in": "に分布し",
        r"grows in":   "で育ち",
        r"common in":  "で一般的に見られ",
        r"widespread in": "に広く分布し",
        r"inhabits":   "に生息し",
        r"occurs in":  "に出現し"
    }
    for en, jp in repl.items():
        text = re.sub(en, jp, text, flags=re.I)
    return text

def get_habitat(label_en):
    try:
        res = wikipedia.search(label_en, results=1)
        if not res: return "生息情報なし"
        summary = wikipedia.page(res[0]).summary
        sents = [s.strip() for s in summary.split('.') if re.search(
            r"\b(native to|found in|endemic to|distributed in|grows in|common in|widespread in|inhabits|occurs in)\b", s, re.I)]
        if not sents: return "生息情報なし"
        jp = simple_translate_habitat("。".join(sents))
        return jp + "。"
    except Exception as e:
        return f"取得失敗: {e}"

def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        e = class_names[torch.argmax(model(t)).item()]
    return en2jp.get(e, e), e

def save_record(img_path, label_jp, habitat):
    dst_dir = os.path.join(ZUKAN_DIR, label_jp)
    os.makedirs(dst_dir, exist_ok=True)
    fn = os.path.basename(img_path)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst_fn = f"{ts}_{fn}"
    shutil.copy(img_path, os.path.join(dst_dir, dst_fn))
    with open(os.path.join(dst_dir, "log.txt"), "a", encoding="utf-8") as f:
        f.write(f"{dst_fn} {ts}\n")
    with open(os.path.join(dst_dir, f"{dst_fn}.meta.txt"), "w", encoding="utf-8") as f:
        f.write(habitat)
    return ts

class ZukanViewer(tk.Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("図鑑一覧")
        self.geometry("600x600")
        c = tk.Canvas(self); sb = tk.Scrollbar(self, orient="vertical", command=c.yview)
        f = tk.Frame(c)
        f.bind("<Configure>", lambda e: c.configure(scrollregion=c.bbox("all")))
        c.create_window((0,0), window=f, anchor="nw")
        c.configure(yscrollcommand=sb.set)
        c.pack(side="left", fill="both", expand=True); sb.pack(side="right", fill="y")
        for lbl in sorted(os.listdir(ZUKAN_DIR)):
            d = os.path.join(ZUKAN_DIR, lbl)
            if not os.path.isdir(d): continue
            logs = {}
            if os.path.exists(os.path.join(d, "log.txt")):
                for ln in open(os.path.join(d, "log.txt"), encoding="utf-8"):
                    fn, ts = ln.strip().split()
                    logs[fn] = ts
            for fn in sorted([f for f in os.listdir(d) if not f.endswith(".txt")]):
                frame = tk.Frame(f, pady=5)
                frame.pack(anchor="w")
                img = Image.open(os.path.join(d, fn)); img.thumbnail((120,120))
                ph = ImageTk.PhotoImage(img)
                lbl_img = tk.Label(frame, image=ph); lbl_img.image = ph
                lbl_img.pack(side="left")
                ts = logs.get(fn, "不明")
                meta = ""
                mp = os.path.join(d, fn + ".meta.txt")
                if os.path.exists(mp):
                    meta = open(mp, encoding="utf-8").read().strip()
                text = f"{lbl}\n日時: {ts}\n生息地: {meta}"
                tk.Label(frame, text=text, justify="left", wraplength=400).pack(side="left", padx=10)

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("花分類＋生息地図鑑")
        self.geometry("450x500")
        tk.Button(self, text="画像を選択", command=self.load).pack(pady=10)
        self.lbl_res = tk.Label(self, text="", wraplength=400, justify="left")
        self.lbl_res.pack(pady=10)
        self.lbl_img = tk.Label(self)
        self.lbl_img.pack()
        tk.Button(self, text="図鑑を開く", command=self.show_zukan).pack(pady=10)

    def load(self):
        p = filedialog.askopenfilename(filetypes=[("画像","*.jpg *.png")])
        if not p: return
        lbljp, lble = predict(p)
        habitat = get_habitat(lble)
        ts = save_record(p, lbljp, habitat)
        self.lbl_res.config(text=f"予測: {lbljp}\n生息地: {habitat}\n取り込み日時: {ts}")
        img = Image.open(p); img.thumbnail((300,300))
        ph = ImageTk.PhotoImage(img)
        self.lbl_img.config(image=ph); self.lbl_img.image=ph
        messagebox.showinfo("完了", f"この画像は「{lbljp}」と分類されました。\n生息地: {habitat}\n{ts} に図鑑保存しました。")

    def show_zukan(self):
        ZukanViewer(self).grab_set()

if __name__ == "__main__":
    App().mainloop()