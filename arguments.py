import argparse
import torch


def get_args():
    parser = argparse.ArgumentParser(
        description='Visual-Language-Navigation')

    # General Arguments
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    
    # Logging, loading models, visualization
    parser.add_argument('--log_interval', type=int, default=10,
                        help="""log interval, one log per n updates
                                (default: 10) """)
    parser.add_argument('-d', '--dump_location', type=str, default="./dump",
                        help='path to dump models and log (default: ./tmp/)')
    parser.add_argument('--exp_name', type=str, default="objectnav-yolo",
                        help='experiment name (default: exp1)')
    parser.add_argument('-v', '--visualize', type=int, default=2,
                        help="""1: Render the observation and
                                   the predicted semantic map,
                                2: Render the observation with semantic
                                   predictions and the predicted semantic map
                                (default: 0)""")
    parser.add_argument('--print_images', type=int, default=0,
                        help='1: save visualization as images')
    parser.add_argument('--save_video', type=int, default=0,
                        help='1: save visualization as video')
    parser.add_argument('-n', '--num_processes', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--gpu_id', type=int, default=0)
    # gui
    parser.add_argument('--path_npz', type=str, default="./saved_pcd/",
                        help='path to saved pcd (default: ./saved_pcd/)')

    
    # Environment, dataset and episode specifications
    parser.add_argument("--task_config", type=str,
                        default="vlobjectnav_hm3d_v2_36sdssadsa.yaml",
                        help="path to config yaml containing task information")
    parser.add_argument("--split", type=str, default="val",
                        help="dataset split (train | val_seen | val_unseen) ")
    parser.add_argument('--episode_count', type=int, default=-1)

    # Model Hyperparameters
    parser.add_argument('--agent', type=str, default="sem_exp")
    parser.add_argument('--num_global_steps', type=int, default=20,
                        help='number of forward steps in A2C (default: 5)')
    parser.add_argument('--turn_angle', type=int, default=30)
    
    # Mapping
    parser.add_argument('-fw', '--frame_width', type=int, default=640,
                        help='Frame width (default:160)')
    parser.add_argument('-fh', '--frame_height', type=int, default=480,
                        help='Frame height (default:120)')
    parser.add_argument('--hfov', type=float, default=79.0,
                        help="horizontal field of view in degrees")
    parser.add_argument('--min_depth', type=float, default=0.0,
                        help="Minimum depth for depth sensor in meters")
    parser.add_argument('--max_depth', type=float, default=5.0,
                        help="Maximum depth for depth sensor in meters")
    parser.add_argument('--map_resolution', type=int, default=5)
    parser.add_argument('--map_size_cm', type=int, default=2400)
    parser.add_argument('--map_height_cm', type=int, default=110)
    parser.add_argument('--collision_threshold', type=float, default=0.10)



    # SAM setting
    parser.add_argument("--box_threshold", type=float, default=0.7)  # 0.3
    parser.add_argument("--text_threshold", type=float, default=0.8)  # 0.25
    parser.add_argument("--nms_threshold", type=float, default=0.5)

    parser.add_argument("--sam_variant", type=str, default="mobilesam",
                        choices=['fastsam', 'mobilesam', "lighthqsam"])
    parser.add_argument("--detector", type=str, default="dino", 
                        choices=["yolo", "dino", "none"], 
                        help="If none, no tagging and detection will be used and the SAM will be run in dense sampling mode. ")
    parser.add_argument("--add_bg_classes", action="store_true", 
                        help="If set, add background classes (wall, floor, ceiling) to the class set. ")
    parser.add_argument("--accumu_classes", action="store_true",
                        help="if set, the class set will be accumulated over frames")



    # LLM setting
    parser.add_argument('--vln_mode', type=str, default="clip",
                        choices=['clip', 'llm', "llm_game"])
    parser.add_argument('--gpt_type', type=int, default=1,
                        help="""0: text-davinci-003
                                1: gpt-3.5-turbo
                                2: gpt-4o
                                (default: 1)""")
                                
    parser.add_argument('--load', type=str, default="0",
                    help="""model path to load,
                            0 to not reload (default: 0)""")
    # parse arguments
    args = parser.parse_args()

    args.cuda = torch.cuda.is_available()

    return args
