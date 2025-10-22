from lib.ai import edit_image

img = edit_image(
    "Extract the man from this graphic design. Return a transparent background PNG.",
    ["https://v3b.fal.media/files/b/rabbit/Wqoen-tYlH1RjfSmV7dcx_1600w-1HZYAUid2AE.webp"],
)

print(img)