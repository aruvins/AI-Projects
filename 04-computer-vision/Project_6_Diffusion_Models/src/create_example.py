from PIL import Image, ImageDraw

# Create blank scene
img = Image.new("RGB", (512, 512), (120, 200, 120))
draw = ImageDraw.Draw(img)

# Draw an oval (simple oval)
draw.ellipse((200, 250, 320, 370), fill=(80, 80, 80))

# Add a sun
draw.ellipse((50, 50, 120, 120), fill=(255, 220, 0))

img.save("data/input.jpg")
print("Saved data/input.jpg")


mask = Image.new("L", (512, 512), 0)  # black = keep
draw = ImageDraw.Draw(mask)

# White circle = area to edit oval
draw.ellipse((200, 250, 320, 370), fill=255)

mask.save("data/mask.png")
print("Saved data/mask.png")