"""
验证 K1 配置文件

这个脚本检查 K1 的配置文件是否正确设置

使用方法:
    python tools/verify_k1_config.py
"""

import os
import sys
import yaml
import xml.etree.ElementTree as ET

# 颜色输出
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def check_file_exists(filepath, description):
    """检查文件是否存在"""
    if os.path.exists(filepath):
        print(f"{GREEN}✓{RESET} {description}: {filepath}")
        return True
    else:
        print(f"{RED}✗{RESET} {description} NOT FOUND: {filepath}")
        return False


def parse_mujoco_xml(xml_path):
    """解析 MuJoCo XML 文件获取关节信息"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    joints = []
    bodies = []
    
    # 递归查找所有关节和body
    def traverse(element):
        if element.tag == 'joint':
            joint_name = element.get('name')
            if joint_name and joint_name != 'world_joint':  # 忽略free joint
                joints.append(joint_name)
        if element.tag == 'body':
            body_name = element.get('name')
            if body_name:
                bodies.append(body_name)
        
        for child in element:
            traverse(child)
    
    traverse(root)
    
    return joints, bodies


def verify_env_config(env_config_path, xml_path):
    """验证环境配置"""
    print(f"\n{BLUE}Checking environment config:{RESET} {env_config_path}")
    
    if not os.path.exists(env_config_path):
        print(f"{RED}✗ Config file not found{RESET}")
        return False
    
    # 读取配置
    with open(env_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    env = config.get('env', {})
    
    # 检查关键字段
    checks = [
        ('char_file', str),
        ('init_pose', list),
        ('key_bodies', list),
        ('contact_bodies', list),
        ('joint_err_w', list),
        ('motion_file', str),
    ]
    
    all_pass = True
    for field, expected_type in checks:
        if field in env:
            value = env[field]
            if isinstance(value, expected_type):
                if field == 'init_pose':
                    print(f"{GREEN}✓{RESET} {field}: {len(value)} dimensions")
                    if len(value) != 28:
                        print(f"{YELLOW}  WARNING: Expected 28 dims (3+3+22), got {len(value)}{RESET}")
                        all_pass = False
                elif field == 'joint_err_w':
                    print(f"{GREEN}✓{RESET} {field}: {len(value)} weights")
                    if len(value) != 22:
                        print(f"{YELLOW}  WARNING: Expected 22 weights, got {len(value)}{RESET}")
                        all_pass = False
                elif isinstance(value, list):
                    print(f"{GREEN}✓{RESET} {field}: {len(value)} items")
                else:
                    print(f"{GREEN}✓{RESET} {field}: {value}")
            else:
                print(f"{RED}✗{RESET} {field}: wrong type (expected {expected_type.__name__})")
                all_pass = False
        else:
            print(f"{RED}✗{RESET} {field}: missing")
            all_pass = False
    
    # 解析 XML 并验证 body 名称
    joints, bodies = parse_mujoco_xml(xml_path)
    
    print(f"\n{BLUE}XML Analysis:{RESET}")
    print(f"  Joints found: {len(joints)}")
    print(f"  Bodies found: {len(bodies)}")
    
    # 验证 key_bodies
    key_bodies = env.get('key_bodies', [])
    print(f"\n{BLUE}Verifying key_bodies:{RESET}")
    for body in key_bodies:
        if body in bodies:
            print(f"{GREEN}✓{RESET} {body}")
        else:
            print(f"{RED}✗{RESET} {body} NOT FOUND in XML")
            all_pass = False
    
    # 验证 contact_bodies
    contact_bodies = env.get('contact_bodies', [])
    print(f"\n{BLUE}Verifying contact_bodies:{RESET}")
    for body in contact_bodies:
        if body in bodies:
            print(f"{GREEN}✓{RESET} {body}")
        else:
            print(f"{RED}✗{RESET} {body} NOT FOUND in XML")
            all_pass = False
    
    return all_pass


def main():
    print(f"{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}K1 Configuration Verification{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 检查文件
    print(f"\n{BLUE}Checking required files:{RESET}")
    
    files_to_check = [
        ('data/assets/k1/K1_serial.xml', 'K1 Model'),
        ('data/envs/deepmimic_k1_env.yaml', 'DeepMimic Env Config'),
        ('data/envs/add_k1_env.yaml', 'ADD Env Config'),
        ('data/envs/amp_k1_env.yaml', 'AMP Env Config'),
        ('data/agents/deepmimic_k1_ppo_agent.yaml', 'DeepMimic Agent'),
        ('data/agents/add_k1_agent.yaml', 'ADD Agent'),
        ('data/agents/amp_k1_agent.yaml', 'AMP Agent'),
        ('args/deepmimic_k1_ppo_args.txt', 'DeepMimic Args'),
        ('args/add_k1_args.txt', 'ADD Args'),
        ('args/amp_k1_args.txt', 'AMP Args'),
    ]
    
    all_exist = True
    for filepath, description in files_to_check:
        full_path = os.path.join(base_dir, filepath)
        if not check_file_exists(full_path, description):
            all_exist = False
    
    # 检查运动数据（可选）
    print(f"\n{BLUE}Checking motion data (optional):{RESET}")
    motion_file = os.path.join(base_dir, 'data/motions/k1/k1_walk.pkl')
    if os.path.exists(motion_file):
        print(f"{GREEN}✓{RESET} Motion file exists: {motion_file}")
    else:
        print(f"{YELLOW}⚠{RESET} Motion file not found: {motion_file}")
        print(f"  {YELLOW}Create one with:{RESET}")
        print(f"    python tools/create_k1_motion_template.py")
    
    # 验证配置
    xml_path = os.path.join(base_dir, 'data/assets/k1/K1_serial.xml')
    
    configs_to_verify = [
        'data/envs/deepmimic_k1_env.yaml',
        'data/envs/add_k1_env.yaml',
        'data/envs/amp_k1_env.yaml',
    ]
    
    all_valid = True
    for config_path in configs_to_verify:
        full_path = os.path.join(base_dir, config_path)
        if os.path.exists(full_path):
            if not verify_env_config(full_path, xml_path):
                all_valid = False
    
    # 总结
    print(f"\n{BLUE}{'='*60}{RESET}")
    if all_exist and all_valid:
        print(f"{GREEN}✅ All checks passed! Ready to train.{RESET}")
        print(f"\n{BLUE}Next steps:{RESET}")
        print(f"1. Create motion data:")
        print(f"   python tools/create_k1_motion_template.py")
        print(f"2. View motion:")
        print(f"   python mimickit/run.py --arg_file args/view_motion_k1_args.txt --visualize true")
        print(f"3. Start training:")
        print(f"   python mimickit/run.py --arg_file args/deepmimic_k1_ppo_args.txt")
    else:
        print(f"{RED}❌ Some checks failed. Please fix the issues above.{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")


if __name__ == '__main__':
    main()


