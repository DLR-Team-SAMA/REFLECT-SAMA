import google.generativeai as genai

import os

import json
def get_robot_plan(folder_name, type_of_summary, step=None, with_obs=False):
    L1_txt_fn = f'{folder_name}/{type_of_summary}_L1.txt'
    L2_txt_fn = f'{folder_name}/{type_of_summary}_L2.txt'
    with open(L2_txt_fn, 'r') as f:
        L2_captions = f.readlines()
    
    with open(L1_txt_fn, 'r') as f:
        L1_captions = f.readlines()

    if with_obs is False:
        captions = L2_captions
    else:
        captions = L1_captions

    robot_plan = ""
    for caption in captions:
        if step is not None and step in caption:
            break
        if with_obs:
            robot_plan += caption
        else:
            robot_plan += caption[:caption.find("Visual observation")-1] + "\n"
    return robot_plan


def run_reasoning(folder_name):
    type_of_summary = "vlm_summary"
    task_json = f'{folder_name}/task.json'
    reasoning_json = f'{folder_name}/reasoning.json'
    prompt_json = 'prompt.json'
    L1_txt_fn = f'{folder_name}/{type_of_summary}_L1.txt'
    L2_txt_fn = f'{folder_name}/{type_of_summary}_L2.txt'
    if not os.path.exists(L1_txt_fn):
        print(f"[INFO] {L1_txt_fn} does not exist. Skipping reasoning.")
        return
    if not os.path.exists(L2_txt_fn):
        print(f"[INFO] {L2_txt_fn} does not exist. Skipping reasoning.")
        return

    GOOGLE_API_KEY=os.getenv('GEMINI_API_KEY')
    genai.configure(api_key=GOOGLE_API_KEY)
    llm = genai.GenerativeModel('gemini-pro')

    with open(task_json) as f:
        task = json.load(f)
    
    if os.path.exists(reasoning_json):
        print("[INFO] Reasoning already generated")
        with open(reasoning_json, 'r') as f:
            reasoning_dict = json.load(f)
        return
    else:
        reasoning_dict = {}

    with open(prompt_json, 'r') as f:
        prompt_info = json.load(f)
    
    # Load L2 captions from state_summary_L2.txt
    with open(L2_txt_fn, 'r') as f:
        L2_captions = f.readlines()

    # Load L1 captions from state_summary_L1.txt
    with open(L1_txt_fn, 'r') as f:
        L1_captions = f.readlines()
    
    # Loop through each subgoal and check for post-condition
    print(">>> Run step-by-step subgoal-level analysis...")
    selected_caption = ""
    prompt = {}

    for caption in L2_captions:
        print(">>> Verify subgoal...")
        subgoal = caption.split(". ")[1].split(": ")[1].lower()

        verifier_system_prompt = prompt_info['subgoal-verifier']['template-system']
        verifier_prompt = prompt_info['subgoal-verifier']['template-user'].replace("[SUBGOAL]", subgoal).replace("[OBSERVATION]", caption[caption.find("Visual observation"):])
        messages = [{'role': 'user', 'parts':[verifier_system_prompt]},
                    {'role': 'model', 'parts':['Understood']},
                    {'role': 'user', 'parts':[verifier_prompt]}]

        response = llm.generate_content(messages)
        print("[INFO] Subgoal verification response:", response.text)
        ans = response.text
        is_success = int(ans.split(", ")[0] == "Yes")
        if is_success == 0:
            selected_caption = caption
            print(f"[INFO] Failure identified in subgoal [{subgoal}] at {caption.split('.')[0]}")
            break
        else:
            print(f"[INFO] Subgoal [{subgoal}] succeeded!")

    if len(selected_caption) != 0:
            print(">>> Get detailed reasoning from L1...")
            step_name = selected_caption.split(".")[0]
            for _, caption in enumerate(L1_captions):
                if step_name in caption:
                    action = caption.split(". ")[1].split(": ")[1].lower()
                    prev_observations = get_robot_plan(folder_name, type_of_summary, step=step_name, with_obs=True)
                    if len(prev_observations) != 0:
                        prompt_name = 'reasoning-execution'
                    else:
                        prompt_name = 'reasoning-execution-no-history'
                    prompt_sys = prompt_info[prompt_name]['template-system']
                    prompt_trg = prompt_info[prompt_name]['template-user'].replace("[ACTION]", action)
                    prompt_trg = prompt_trg.replace("[TASK_NAME]", task['name'])
                    prompt_trg = prompt_trg.replace("[STEP]", step_name)
                    prompt_trg = prompt_trg.replace("[SUMMARY]", prev_observations)
                    prompt_trg = prompt_trg.replace("[OBSERVATION]", caption[caption.find("Action"):])

                    messages = []
                    messages.append({'role': 'user', 'parts':[prompt_sys]})
                    messages.append({'role': 'model', 'parts':['Understood']})
                    messages.append({'role': 'user', 'parts':[prompt_trg]})

                    response = llm.generate_content(messages)
                    ans = response.text


                    print("[INFO] Predicted failure reason:", ans)
                    reasoning_dict['pred_failure_reason'] = ans

                    prompt = {}
                    prompt_sys = prompt_info['reasoning-execution-steps']['template-system']
                    prompt_trg = prompt_info['reasoning-execution-steps']['template-user'].replace("[FAILURE_REASON]", ans)

                    messages = []
                    messages.append({'role': 'user', 'parts':[prompt_sys]})
                    messages.append({'role': 'model', 'parts':['Understood']})
                    messages.append({'role': 'user', 'parts':[prompt_trg]})
                    response = llm.generate_content(messages)
                    time_steps = response.text
                    
                    print("[INFO] Predicted failure time steps:", time_steps, time_steps.split(", "))
                    reasoning_dict['pred_failure_step'] = [time_step.replace(",", "") for time_step in time_steps.split(", ")]
                    break
    else:
        print(">>> All actions are executed successfully, run plan-level analysis...")

        prompt_sys = prompt_info['reasoning-plan']['template-system']
        prompt_dsr = prompt_info['reasoning-plan']['template-user'].replace("[TASK_NAME]", task['name'])
        prompt_dsr = prompt_dsr.replace("[SUCCESS_CONDITION]", task['success_condition'])
        prompt_dsr = prompt_dsr.replace("[CURRENT_STATE]", L1_captions[-1].split(". ")[1].split(": ")[1])
        prompt_dsr = prompt_dsr.replace("[OBSERVATION]", get_robot_plan(folder_name, step=None, with_obs=False))
        messages = [{'role': 'user', 'parts':[prompt_sys]},
                    {'role': 'model', 'parts':['Understood']},
                    {'role': 'user', 'parts':[prompt_dsr]}]
        response = llm.generate_content(messages)
        ans = response.text

        
        print("[INFO] Predicted failure reason:", ans)
        reasoning_dict['pred_failure_reason'] = ans

        prompt_sys = prompt_info['reasoning-plan-steps']['template-system']
        prompt_dsr = prompt_info['reasoning-plan-steps']['template-user'].replace("[PREV_PROMPT]", prompt_dsr + " " + ans)
        messages = [{'role': 'user', 'parts':[prompt_sys]},
                    {'role': 'model', 'parts':['Understood']},
                    {'role': 'user', 'parts':[prompt_dsr]}]
        response = llm.generate_content(messages)
        step = response.text
        step_str = step.split(" ")[0]
        if step_str[-1] == '.' or step_str[-1] == ',':
            step_str = step_str[:-1]

        print("[INFO] Predicted failure time steps:", step_str)
        reasoning_dict['pred_failure_step'] = step_str

    reasoning_dict['gt_failure_reason'] = task['gt_failure_reason']
    reasoning_dict['gt_failure_step'] = task['gt_failure_step']
    
    with open(reasoning_json, 'w') as f:
        json.dump(reasoning_dict, f)


# folder = 'reflect/main/keyframe_dataset/boilWater/boilWater-1'
keyframe_dataset = 'reflect/main/keyframe_dataset'
tasks = os.listdir(keyframe_dataset)
for task in tasks:
    cases = os.listdir(f'{keyframe_dataset}/{task}')
    for case in cases:
        folder = f'{keyframe_dataset}/{task}/{case}'
        print(f'Running reasoning for {folder}...')
        run_reasoning(folder)