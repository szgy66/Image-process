import cv2
import numpy as np
from time import time

img1 = cv2.imread('D:/Desktop/Y_HR/0_6.png', 0)  # HR

img2 = cv2.imread('D:/Desktop/Y_LR/0_6.png', 0)  # LR
img2 = img2[375:645, 740:1220]
img2 = cv2.resize(img2, dsize=None, fx=4, fy=4)


akaze = cv2.AKAZE_create()
kp1, descriptor1 = akaze.detectAndCompute(img1, None)
kp2, descriptor2 = akaze.detectAndCompute(img2, None)
bf = cv2.BFMatcher()
matches = bf.knnMatch(descriptor1, descriptor2, k=2)

good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append([m])

ref_matched_kpts = np.float32([kp1[m[0].queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
sensed_matched_kpts = np.float32([kp2[m[0].trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

result = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite('D:/Desktop/re.png', result)

H, status = cv2.findHomography(ref_matched_kpts, sensed_matched_kpts, cv2.RANSAC, 5.0)
warped_image = cv2.warpPerspective(img1, H, (img2.shape[1], img2.shape[0]))  # 返回高质量图像经过单映性变换后的结果

H, W = img2.shape

mask = np.zeros((H, W), dtype=np.uint8)

patch = 40  # 替换的ROI大小
alph = 4  # 搜索范围
thre = 0
stride = 40  # ROI滑动时的步长
lens = 6
substract = 0.4
corr = np.zeros((H, W), dtype=np.uint8)

def remove(ls):
    for i in range(len(ls)):
        if np.isnan(ls[i]) == True:
            ls[i] = 0
    return ls

def select_roi(m, n):
    hr_roi = img1[m: m + patch, n:n + patch]
    hr_roi = hr_roi.reshape(hr_roi.size, order='C')
    correlation1 = np.corrcoef(lr_roi, hr_roi).sum()
    return correlation1

def judge(ls, local_site, temp):
    ls = remove(ls)
    new_ls = ls[ls.index(np.max(ls)):]
    if all(x > y for x, y in zip(new_ls, new_ls[1:])) == True and len(new_ls) > lens:
        mask[i:i + patch, j:j + patch] = img1[local_site[0]:local_site[0] + patch, local_site[1]:local_site[1] + patch]
        corr[i:i + patch, j:j + patch] = (temp / 4 * 255).astype(np.uint8)
    else:
        temp = cor1
        local_site = [i, j]
        mask[i:i + patch, j:j + patch] = img1[local_site[0]:local_site[0] + patch, local_site[1]:local_site[1] + patch]
        corr[i:i + patch, j:j + patch] = (temp / 4 * 255).astype(np.uint8)

start = time()
for i in range(0, H - patch + 1, stride):
    for j in range(0, W - patch + 1, stride):
        lr_roi = img2[i:i + patch, j:j + patch]
        lr_roi = lr_roi.reshape(lr_roi.size, order='C')
        local_site = [0, 0]
        temp = 0
        cor1 = select_roi(i, j)
        ls = []
        if alph <= i < H - patch - alph and alph <= j < W - patch - alph:
            for n in range(j - alph, j + alph):
                cor2 = select_roi(i, n)
                ls.append(cor2)
                if cor2 > temp:
                    temp = cor2
                    local_site = [i, n]
            judge(ls, local_site, temp)
            continue

        elif i < alph and j < alph:
            for n in range(0, j + alph):
                cor2 = select_roi(i, n)
                ls.append(cor2)
                if cor2 > temp:
                    temp = cor2
                    local_site = [i, n]
            judge(ls, local_site, temp)
            continue

        elif i < alph and alph <= j < W - patch - alph:
            for n in range(j - alph, j + alph):
                cor2 = select_roi(i, n)
                ls.append(cor2)
                if cor2 > temp:
                    temp = cor2
                    local_site = [i, n]
            judge(ls, local_site, temp)
            continue

        elif i < alph and W - patch - alph <= j:
            for n in range(j - alph, W - patch):
                cor2 = select_roi(i, n)
                ls.append(cor2)
                if cor2 > temp:
                    temp = cor2
                    local_site = [i, n]
            judge(ls, local_site, temp)
            continue

        elif j < alph and alph <= i < H - patch - alph:
            for n in range(0, j + alph):
                cor2 = select_roi(i, n)
                ls.append(cor2)
                if cor2 > temp:
                    temp = cor2
                    local_site = [i, n]
            judge(ls, local_site, temp)
            continue

        elif j >= W - patch - alph and alph <= i < H - patch - alph:
            for n in range(j - alph, W - patch):
                cor2 = select_roi(i, n)
                ls.append(cor2)
                if cor2 > temp:
                    temp = cor2
                    local_site = [i, n]
            judge(ls, local_site, temp)
            continue

        elif i >= H - patch - alph and j < alph:
            for n in range(0, j + alph):
                cor2 = select_roi(i, n)
                ls.append(cor2)
                if cor2 > temp:
                    temp = cor2
                    local_site = [i, n]
            judge(ls, local_site, temp)
            continue

        elif i >= H - patch - alph and alph <= j < W - alph - patch:
            for n in range(j - alph, j + alph):
                cor2 = select_roi(i, n)
                ls.append(cor2)
                if cor2 > temp:
                    temp = cor2
                    local_site = [i, n]
            judge(ls, local_site, temp)
            continue

        elif i >= H - patch - alph and j >= W - patch - alph:
            for n in range(j - alph, W - patch):
                cor2 = select_roi(i, n)
                ls.append(cor2)
                if cor2 > temp:
                    temp = cor2
                    local_site = [i, n]
            judge(ls, local_site, temp)
    print("Now, it is %d rows." % i)
end = time()
print('It spend %f s' % (end - start))

cv2.imwrite('D:/Desktop/mask3.png', mask)  # 保存模板匹配的结果

cor = np.zeros((H, W), dtype=np.uint8)
for i in range(H-40):
    for j in range(W-40):
        roi1 = mask[i:i+40, j:j+40]
        roi1 = roi1.reshape(roi1.size, order='C')
        roi2 = img2[i:i+40, j:j+40]
        roi2 = roi2.reshape(roi2.size, order='C')
        corr = np.corrcoef(roi1, roi2).sum()
        cor[i:i+40, j:j+40] = corr
cor = (cor/4*255).astype(np.uint8)
mean = np.mean(cor)
ret, th = cv2.threshold(cor, mean, 255, cv2.THRESH_BINARY)
cv2.imwrite('D:/Desktop/cor.png', th)