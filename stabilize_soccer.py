import cv2
import numpy as np

# ==========================
# フレーム読み込み
# ==========================
img1 = cv2.imread("frames/frame_000100.png")
img2 = cv2.imread("frames/frame_000101.png")

if img1 is None or img2 is None:
    print("画像が読み込めません")
    exit()

# ==========================
# グレースケール変換
# ==========================
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# ==========================
# ORB特徴点抽出
# ==========================
orb = cv2.ORB_create(3000)

kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

if des1 is None or des2 is None:
    print("特徴点が見つかりません")
    exit()

# ==========================
# 特徴点マッチング
# ==========================
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# 上位500点のみ使用
matches = matches[:500]

# ==========================
# 対応点取得
# ==========================
pts1 = np.float32(
    [kp1[m.queryIdx].pt for m in matches]
).reshape(-1, 1, 2)

pts2 = np.float32(
    [kp2[m.trainIdx].pt for m in matches]
).reshape(-1, 1, 2)

# ==========================
# Homography推定
# ==========================
H, mask = cv2.findHomography(
    pts2,
    pts1,
    cv2.RANSAC,
    5.0
)

print("Homography Matrix")
print(H)

# ==========================
# 補正
# ==========================
h, w = img1.shape[:2]

stabilized = cv2.warpPerspective(
    img2,
    H,
    (w, h)
)

# ==========================
# 差分画像
# ==========================
diff_before = cv2.absdiff(img1, img2)
diff_after = cv2.absdiff(img1, stabilized)

# ==========================
# マッチング可視化
# ==========================
match_img = cv2.drawMatches(
    img1,
    kp1,
    img2,
    kp2,
    matches[:50],
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# ==========================
# 表示
# ==========================
cv2.imshow("Frame 100", img1)
cv2.imshow("Frame 101", img2)
cv2.imshow("Stabilized", stabilized)

cv2.imshow("Diff Before", diff_before)
cv2.imshow("Diff After", diff_after)

cv2.imshow("ORB Matches", match_img)

cv2.waitKey(0)
cv2.destroyAllWindows()