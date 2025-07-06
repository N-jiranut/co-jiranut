import math

def angle360(u, v):
    # u, v คือ tuple หรือ list เช่น [ux, uy]
    ux, uy = u
    vx, vy = v
    # dot product และ cross product แบบ 2D (ผลเฉพาะ componnent z)
    dot = ux * vx + uy * vy
    cross = ux * vy - uy * vx
    # คำนวณมุมด้วย atan2(cross, dot) → ได้มุมในช่วง -π..π
    theta = math.atan2(cross, dot)
    # แปลงเป็นองศา
    deg = math.degrees(theta)
    # ถ้าเป็นลบ ให้อัปค่า +360 เพื่อให้อยู่ในช่วง 0–360
    return deg + 360 if deg < 0 else deg

# ทดลองใช้
a = (3.75, 2)
b = (0, 1)
print(angle360(a, b))