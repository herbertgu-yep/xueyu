#!/bin/bash
# 学屿 · macOS 客户端打包（PyInstaller .app + hdiutil .dmg）
# 用法：./scripts/build_mac.sh
# 产物：dist/学屿.app 与 dist/学屿-<版本>.dmg
set -euo pipefail
cd "$(dirname "$0")/.."

VERSION=$(python3 -c "import re; print(re.search(r'version = \"([^\"]+)\"', pyproject.toml).group(1))")
echo "=== 学屿 v$VERSION 打包 ==="

# 1) 依赖
if ! .venv/bin/python -c "import PyInstaller, PIL" 2>/dev/null; then
  echo "安装打包工具（pyinstaller / pillow）…"
  uv pip install --python .venv/bin/python pyinstaller pillow
fi

# 2) 图标（不存在时生成）
if [ ! -f assets/icon.icns ]; then
  echo "生成应用图标…"
  .venv/bin/python - <<'EOF'
from PIL import Image, ImageDraw
import subprocess, os

S = 1024
img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
d = ImageDraw.Draw(img)
d.rounded_rectangle([0, 0, S, S], radius=220, fill=(250, 244, 230, 255))
d.rectangle([40, S // 2 - 20, S - 40, S - 40], fill=(125, 184, 247, 255))
for y in range(S // 2 - 20, S - 40):
    t = (y - (S // 2 - 20)) / (S // 2 - 60)
    d.line([40, y, S - 40, y], fill=(int(125 + t * 40), int(184 - t * 40), int(247 - t * 60), 255))
mask = Image.new("L", (S, S), 0)
ImageDraw.Draw(mask).rounded_rectangle([0, 0, S, S], radius=220, fill=255)
sea = Image.new("RGBA", (S, S), (0, 0, 0, 0))
sea.paste(img, (0, 0), mask)
img = sea
d = ImageDraw.Draw(img)
d.ellipse([S - 300, 90, S - 160, 230], fill=(240, 180, 41, 255), outline=(61, 55, 44, 255), width=10)
d.ellipse([120, 140, 340, 220], fill=(255, 255, 255, 230), outline=(61, 55, 44, 255), width=8)
d.ellipse([200, 100, 400, 180], fill=(255, 255, 255, 230), outline=(61, 55, 44, 255), width=8)
d.ellipse([200, 560, 824, 760], fill=(238, 205, 148, 255), outline=(61, 55, 44, 255), width=12)
d.ellipse([260, 520, 764, 700], fill=(126, 211, 165, 255), outline=(61, 55, 44, 255), width=12)
import math
trunk_x, trunk_y = 500, 640
d.line([trunk_x, trunk_y, trunk_x - 30, trunk_y - 180], fill=(139, 94, 60, 255), width=26)
for ang in (-150, -120, -60, -30, 200, 230):
    rad = math.radians(ang)
    d.line([trunk_x - 30, trunk_y - 180,
            trunk_x - 30 + math.cos(rad) * 170, trunk_y - 180 + math.sin(rad) * 110],
           fill=(76, 175, 125, 255), width=22)
d.ellipse([trunk_x - 44, trunk_y - 196, trunk_x - 16, trunk_y - 168], fill=(240, 180, 41, 255))
d.polygon([(360, 40), (664, 40), (644, 120), (340, 120)], fill=(242, 123, 167, 210))
d.rounded_rectangle([0, 0, S, S], radius=220, outline=(61, 55, 44, 255), width=22)
os.makedirs("assets", exist_ok=True)
img.save("assets/icon.png")
EOF
  mkdir -p /tmp/xueyu.iconset
  for sz in 16 32 64 128 256 512; do
    sips -z $sz $sz assets/icon.png --out /tmp/xueyu.iconset/icon_${sz}x${sz}.png > /dev/null
    sips -z $((sz*2)) $((sz*2)) assets/icon.png --out /tmp/xueyu.iconset/icon_${sz}x${sz}@2x.png > /dev/null
  done
  iconutil -c icns /tmp/xueyu.iconset -o assets/icon.icns
  rm -rf /tmp/xueyu.iconset
  echo "图标已生成 assets/icon.icns"
fi

# 3) PyInstaller 构建
echo "PyInstaller 构建（首次约 3-6 分钟）…"
.venv/bin/python -m PyInstaller xueyu.spec --noconfirm

# 4) DMG
echo "制作 DMG…"
rm -rf /tmp/dmg_stage "dist/学屿-$VERSION.dmg"
mkdir -p /tmp/dmg_stage
cp -R "dist/学屿.app" /tmp/dmg_stage/
ln -s /Applications /tmp/dmg_stage/Applications
hdiutil create -volname "学屿" -srcfolder /tmp/dmg_stage -ov -format UDZO "dist/学屿-$VERSION.dmg" > /dev/null
rm -rf /tmp/dmg_stage

echo ""
echo "=== 打包完成 ==="
ls -lh "dist/学屿.app" "dist/学屿-$VERSION.dmg" | awk '{print $5, $9}'
echo "提示：本机构建为临时签名，双击即可运行；分发给他人前建议 codesign + notarize。"
