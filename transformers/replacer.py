from sys import argv
from os import system


def main():
    if len(argv) > 4:
        print("missing config file or input file")
        print("<config file> <python file> <sh file>")
        return

    try:
        config_file = open(argv[1], "r", encoding='utf-8')
    except OSError:
        print("could not open config file")
        return

    try:
        python_file = open(argv[2], "r", encoding='utf-8')
    except OSError:
        print("could not open python file")
        return

    try:
        sh_file = open(argv[3], "r", encoding='utf-8')
    except OSError:
        print("could not open sh file")
        return

    i = 0
    for config in get_configs(config_file):
        sh_file.seek(0)
        python_file.seek(0)
        create_and_run(config, python_file, sh_file, i)
        i += 1

    python_file.close()
    sh_file.close()


def create_and_run(config, python_file, sh_file, i):
    python_text = python_file.read()
    for replace in config["py"]:
        subject = replace[0]
        replacements = replace[1]
        for replacement in replacements:
            if replacement == "null":
                repl_index = subject.find("!")
                replacement = str(i)
            else:
                repl_index = subject.find("?")

            beg_index = python_text.find(subject[0:repl_index])

            python_text = python_text[0:(beg_index + repl_index)] + replacement + python_text[(beg_index + repl_index + 1):]
            subject = subject[0:repl_index] + replacement + subject[(repl_index + 1):]

    with open("network_" + str(i) + ".py", "w+", encoding='utf-8') as of:
        of.write(python_text)

    sh_text = sh_file.read()
    for replace in config["sh"]:
        subject = replace[0]
        replacements = replace[1]
        for replacement in replacements:
            if replacement == "null":
                repl_index = subject.find("!")
                replacement = str(i)
            else:
                repl_index = subject.find("?")

            beg_index = sh_text.find(subject[0:repl_index])

            sh_text = sh_text[0:(beg_index + repl_index)] + replacement + sh_text[(beg_index + repl_index + 1):]
            subject = subject[0:repl_index] + replacement + subject[(repl_index + 1):]

    with open("BERT_" + str(i) + ".sh", "w+", encoding='utf-8') as of:
        of.write(sh_text)

    system("qsub BERT_" + str(i) + ".sh")


def get_configs(config_file):
    config_lines = config_file.readlines()
    config_file.close()

    in_py = False
    in_sh = False
    sh = []
    py = []
    configs = []
    for line in config_lines:
        if "#py" in line:
            in_py = True
            if in_sh:
                in_sh = False
                configs.append({"sh": sh.copy(), "py": py.copy()})
                py = []
                sh = []

        elif "#sh" in line:
            in_py = False
            in_sh = True
        elif in_py:
            subject, replacements = parse_line(line)
            py.append((subject, replacements))
        elif in_sh:
            subject, replacements = parse_line(line)
            sh.append((subject, replacements))

    configs.append({"sh": sh.copy(), "py": py.copy()})

    return configs


def parse_line(line: str):
    split = line.split("::")[:-1]
    subject = split[0]
    replacements = split[1:]

    return subject, replacements






main()
