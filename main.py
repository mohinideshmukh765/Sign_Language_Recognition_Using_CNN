import cv2
import tkinter as tk
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from src.utils.camera_utils import initialize_camera, release_camera
from src.real_time_preprocessing import preprocess_frame
from app import launch_app

# Initialize the main Tkinter window
root = tk.Tk()
root.title("Sign Language Recognition")

# Initially maximize the window with a title bar
root.state("zoomed")

# Load the background image
background_image_path = "C:/VSCode/project/src/imgbg.jpeg"
background_image = Image.open(background_image_path)

# Convert the image to ImageTk format
background_photo = ImageTk.PhotoImage(background_image)

# Create a label to display the background image
background_label = tk.Label(root, image=background_photo)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

# Function to resize the background image dynamically
def resize_background(event):
    new_width, new_height = event.width, event.height
    resized_image = background_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    new_background_photo = ImageTk.PhotoImage(resized_image)
    background_label.config(image=new_background_photo)
    background_label.image = new_background_photo

# Bind the resize event
root.bind("<Configure>", resize_background)

# Label to show camera feed placeholder
camera_label = tk.Label(root)
camera_label.place(x=400, y=200)

# Start camera feed function
def start_camera():
    root.destroy()
    launch_app()

# Exit the application function
def exit_application():
    root.quit()

# Toggle fullscreen layout with five images
def toggle_signs_video():
    signs_window = tk.Toplevel(root)
    signs_window.attributes("-fullscreen", True)
    signs_window.configure(bg="white")
    
    # Load and display images (replace with your image paths)
    images = [
        "C:/VSCode/project/src/a-f.jpeg",
        "C:/VSCode/project/src/g-l.jpeg",
        "C:/VSCode/project/src/m-r.jpeg",
        "C:/VSCode/project/src/s-x.jpeg",
        "C:/VSCode/project/src/y-z.jpeg",
        "C:/VSCode/project/src/1-9.jpeg"
    ]
    photo_images = [ImageTk.PhotoImage(Image.open(img).resize((300, 300), Image.Resampling.LANCZOS)) for img in images]
    
    # Place images in grid
    tk.Label(signs_window, image=photo_images[0], bg="white").grid(row=0, column=0, padx=10, pady=10)
    tk.Label(signs_window, image=photo_images[1], bg="white").grid(row=0, column=1, padx=10, pady=10)
    tk.Label(signs_window, image=photo_images[2], bg="white").grid(row=1, column=0, padx=10, pady=10)
    tk.Label(signs_window, image=photo_images[3], bg="white").grid(row=1, column=1, padx=10, pady=10)
    tk.Label(signs_window, image=photo_images[4], bg="white").grid(row=0, column=2, columnspan=2, pady=20)
    tk.Label(signs_window, image=photo_images[5], bg="white").grid(row=1, column=2, rowspan=2,padx=20, pady=10)
    
    # Keep references to avoid garbage collection
    signs_window.images = photo_images
    
    # Add a Back button in row 3, column 0
    def close_signs_window():
        signs_window.destroy()

    back_button = tk.Button(signs_window, text="Back", command=close_signs_window, font=("Arial", 14, "bold"), 
                            bg="#4A90E2", fg="white", relief="flat", width=10, height=2)
    back_button.grid(row=0, column=4, pady=20, padx=10)
    
    # Add Start Camera button next to the Back button
    start_button = tk.Button(signs_window, text="Start Camera", command=start_camera, font=("Arial", 14, "bold"),
                             bg="#FF66B2", fg="white", relief="flat", width=15, height=2)
    start_button.grid(row=0, column=5, pady=20, padx=10)

    # Escape key to exit fullscreen
    signs_window.bind("<Escape>", lambda e: signs_window.destroy())

# Function to create buttons with hover effect
def create_button(text, command, color, hover_color):
    button = tk.Button(root, text=text, command=command, font=("Arial", 14, "bold"), 
                       bg=color, fg="white", relief="flat", width=20, height=2)
    
    def on_enter(e):
        button.config(bg=hover_color)

    def on_leave(e):
        button.config(bg=color)

    button.bind("<Enter>", on_enter)
    button.bind("<Leave>", on_leave)
    return button

# Create buttons
start_button = create_button("Start Camera", start_camera, "#4A90E2", "#357ABD")
start_button.place(x=410, y=320)  # Fixed position for Start Camera button

signs_button = create_button("Signs", toggle_signs_video, "#4A90E2", "#357ABD")
signs_button.place(x=410, y=410)  # Fixed position for Signs button

exit_button = create_button("Exit", exit_application, "#4A90E2", "#357ABD")
exit_button.place(x=410, y=500)  # Fixed position for Exit button

# Start the Tkinter main loop
root.mainloop()