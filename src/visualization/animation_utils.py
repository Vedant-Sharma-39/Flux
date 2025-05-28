# microbial_colony_sim/src/visualization/animation_utils.py
# Placeholder for animation utilities (e.g., using matplotlib.animation.FuncAnimation
# or saving frames to be compiled into a video with ffmpeg).


class AnimationUtils:
    def __init__(self, visualizer, config):
        self.visualizer = visualizer  # The ColonyVisualizer instance
        self.config = config
        self.frame_count = 0

    def save_animation_frame(self, current_time):
        # Uses the visualizer to plot a state and saves it as a numbered frame
        # self.visualizer.plot_colony_state(current_time, self.frame_count)
        # The plot_colony_state already saves, so this might just coordinate filenames
        # or directly call the savefig part.
        self.frame_count += 1
        pass

    def compile_frames_to_video(self, output_video_filename="colony_animation.mp4"):
        # Uses a tool like ffmpeg to compile saved frames into a video.
        # Requires ffmpeg to be installed and accessible.
        pass
