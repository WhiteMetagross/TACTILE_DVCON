import os
import sys
from pathlib import Path
from typing import Optional

# Suppress annoying Qt font warnings in WSL when cv2.imshow runs
os.environ['QT_LOGGING_RULES'] = 'qt.qpa.fonts.warning=false;qt.qpa.fonts=false'
os.environ['QT_QPA_FONTDIR'] = '/usr/share/fonts'

sys.path.insert(0, str(Path(__file__).parent))

from Tactile.Config.Tasks import TASK_NAMES, NUM_TASKS
from Tactile.Inference import TACTILEPipeline
from Demo import draw_result, find_test_images

class InteractiveCli:
    def __init__(self, DataDirectory: str = "./CocoTaskDataset", OutputDirectory: str = "./output"):
        self.DataDirectory = DataDirectory
        self.OutputDirectory = OutputDirectory
        self.Pipeline = None

    def InitializePipeline(self) -> None:
        print("Initializing TACTILE Pipeline...")
        WeightsDirectory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Tactile", "Weights")
        SamPath = os.path.join(WeightsDirectory, "SamIp.pth")
        EmbPath = os.path.join(WeightsDirectory, "TaskEmbeddings.npy")
        PriorPath = os.path.join(WeightsDirectory, "ClassTaskPrior.npy")
        
        # Determine device
        try:
            import torch
            Device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            Device = "cpu"
            
        self.Pipeline = TACTILEPipeline(
            device=Device,
            sam_weights_path=SamPath if os.path.exists(SamPath) else None,
            task_emb_path=EmbPath if os.path.exists(EmbPath) else None,
            prior_table_path=PriorPath if os.path.exists(PriorPath) else None,
        )
        print("Pipeline initialization complete.\n")

    def DisplayTasks(self) -> None:
        print("\nAvailable Tasks:")
        for TaskId in range(NUM_TASKS):
            print(f"[{TaskId}] {TASK_NAMES[TaskId]}")
            
    def SelectTask(self) -> Optional[int]:
        self.DisplayTasks()
        while True:
            try:
                Choice = input("\nEnter Task ID (or 'q' to quit): ")
                if Choice.lower() == 'q':
                    return None
                TaskId = int(Choice)
                if 0 <= TaskId < NUM_TASKS:
                    return TaskId
                print(f"Invalid Task ID. Please enter a number between 0 and {NUM_TASKS - 1}.")
            except ValueError:
                print("Invalid input. Please enter a valid number.")

    def SelectImage(self, TaskId: int) -> Optional[str]:
        print(f"\nSearching for images related to task: {TASK_NAMES[TaskId]}...")
        Images = find_test_images(self.DataDirectory, TaskId, max_images=10)
        
        if not Images:
            print("No images found for this task in the dataset.")
            while True:
                ManualPath = input("Enter full path to an image (or 'b' to go back): ")
                if ManualPath.lower() == 'b':
                    return None
                if os.path.exists(ManualPath):
                    return ManualPath
                print("Image not found.")

        print("\nAvailable Images:")
        for Index, ImagePath in enumerate(Images):
            print(f"[{Index}] {os.path.basename(ImagePath)}")
            
        DefaultImageDir = "/mnt/c/Users/Xeron/Desktop/DVCONImplementation/CocoTaskDataset/Coco/val2017/"
        
        while True:
            try:
                Choice = input(f"\nEnter Image ID (0-{len(Images)-1}), or 'm' to enter filename, 'b' to go back: ")
                if Choice.lower() == 'b':
                    return None
                if Choice.lower() == 'm':
                    ManualName = input(f"Enter image filename (e.g., 8211 or 000000008211.jpg): ")
                    
                    # Assume it's in the default val2017 dir if not an absolute path
                    if not os.path.isabs(ManualName):
                        # Auto-pad and append .jpg if user just typed the raw number in 'm' mode
                        if ManualName.isdigit():
                            ManualName = f"{int(ManualName):012d}.jpg"
                        ManualPath = os.path.join(DefaultImageDir, ManualName)
                    else:
                        ManualPath = ManualName
                        
                    # Quick check if it exists (handles WSL path)
                    if os.path.exists(ManualPath):
                        return ManualPath
                    
                    # Fallback check relative to Windows path if running outside WSL
                    WinFallback = os.path.join(self.DataDirectory, "Coco", "val2017", ManualName)
                    if os.path.exists(WinFallback):
                        return WinFallback
                        
                    print(f"Image not found. Looked in: {ManualPath}")
                    continue
                
                ImageId = int(Choice)
                if 0 <= ImageId < len(Images):
                    return Images[ImageId]
                
                # If the number is outside the list index, assume it's a direct COCO image ID!
                CocoFilename = f"{ImageId:012d}.jpg"
                CocoPath = os.path.join(DefaultImageDir, CocoFilename)
                if os.path.exists(CocoPath):
                    return CocoPath
                    
                WinFallback = os.path.join(self.DataDirectory, "Coco", "val2017", CocoFilename)
                if os.path.exists(WinFallback):
                    return WinFallback
                    
                print(f"Invalid Image ID. Also could not find COCO dataset image for ID: {CocoFilename}")
            except ValueError:
                print("Invalid input.")

    def RunInference(self, TaskId: int, ImagePath: str) -> None:
        print(f"\nRunning Inference on {os.path.basename(ImagePath)} for task '{TASK_NAMES[TaskId]}'...")
        
        Result = self.Pipeline.infer(ImagePath, TaskId, verbose=True)
        
        os.makedirs(self.OutputDirectory, exist_ok=True)
        OutputPath = os.path.join(self.OutputDirectory, f"cli_result_{os.path.basename(ImagePath)}")
        
        import cv2
        draw_result(ImagePath, Result, OutputPath)
        print(f"\nVisualization saved to {OutputPath}")
        print("Opening live visual. Press ANY key on the image window to continue...")
        
        # Display the image live using OpenCV
        img = cv2.imread(OutputPath)
        if img is not None:
            WindowName = f"TACTILE - Task {TaskId} ({TASK_NAMES[TaskId]})"
            cv2.imshow(WindowName, img)
            cv2.waitKey(0)  # Wait indefinitely until the user presses a key
            cv2.destroyAllWindows()
        else:
            print("Failed to load image for OpenCV display.")

    def Run(self) -> None:
        print("=" * 60)
        print("TACTILE Interactive CLI")
        print("=" * 60)
        
        self.InitializePipeline()
        
        while True:
            TaskId = self.SelectTask()
            if TaskId is None:
                break
                
            while True:
                ImagePath = self.SelectImage(TaskId)
                if ImagePath is None:
                    break
                    
                self.RunInference(TaskId, ImagePath)
                
                Continue = input("\nTest another image for this task? (y/n): ")
                if Continue.lower() != 'y':
                    break
                    
        print("\nExiting TACTILE CLI. Goodbye!")

def Main():
    Cli = InteractiveCli()
    Cli.Run()

if __name__ == "__main__":
    Main()
