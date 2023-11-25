def convert_conda_requirements_to_pip(conda_req_file, pip_req_file):
    def read_file(file_path, encoding):
        with open(file_path, 'r', encoding=encoding) as file:
            return file.readlines()

    try:
        conda_requirements = read_file(conda_req_file, 'utf-8')
    except UnicodeDecodeError:
        conda_requirements = read_file(conda_req_file, 'utf-16')

    pip_requirements = []
    for line in conda_requirements:
        line = line.replace('\x00', '').strip()
        parts = line.split('=')
        if len(parts) >= 2:
            package, version = parts[0], parts[1]
            pip_requirements.append(f"{package}=={version}\n")

    with open(pip_req_file, 'w', encoding='utf-8') as file:
        file.writelines(pip_requirements)

# Example usage
convert_conda_requirements_to_pip('requirements.txt', 'requirements_pip.txt')
