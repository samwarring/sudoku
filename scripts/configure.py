import argparse
import datetime
import os
import platform
import re
import subprocess
import sys


def main():
    # Parse command line
    parser = argparse.ArgumentParser(description='Configure C++ compilation environment')
    parser.add_argument('root_directory')
    parser.add_argument('build_directory')
    parser.add_argument('-c', '--config', choices=['Debug', 'Release', 'RelWithDebInfo', 'MinSizeRel'])
    parser.add_argument('-g', '--coverage', choices=['ON', 'OFF'], default='OFF')
    parser.add_argument('-e', '--ci-env', choices=['appveyor', 'travis'])
    args = parser.parse_args()
    args.coverage = (args.coverage == 'ON')

    # Get relative path to root_directory from build_directory
    root_relative = os.path.relpath(args.root_directory, args.build_directory)

    # Make CMake command line
    cmd = ['cmake']
    all_vars = {}
    all_vars.update(get_config_vars(args.config))
    all_vars.update(get_boost_vars())
    version_vars = get_version_vars()
    all_vars.update(version_vars)
    all_vars.update(get_coverage_vars(args.coverage))
    for (argname, argval) in all_vars.items():
        cmd.append('-D' + argname + '=' + argval)
    cmd.append(root_relative)

    # Appveyor needs to update build version
    if args.ci_env == 'appveyor':
        check_output(['appveyor', 'UpdateBuild', '-Version', version_vars['sudoku_VERSION_DESC']])

    # Make and move into build directory
    mkdir(args.build_directory)
    chdir(args.build_directory)

    # Configure with CMake
    print_cmd(cmd)
    subprocess.check_call(cmd)

    # No errors
    return 0


def quote(s):
    if ' ' in s:
        return '"' + s + '"'
    else:
        return s


def print_cmd(cmdline):
    if isinstance(cmdline, str):
        print('+ ' + str(cmdline))
    elif isinstance(cmdline, list):
        print('+ ' + ' '.join(map(quote, cmdline)))


def check_output(*args, **kwargs):
    print_cmd(args[0])
    return subprocess.check_output(*args, **kwargs).decode('utf-8').strip()


def mkdir(dirname):
    print_cmd(['mkdir', dirname])
    if not os.path.isdir(dirname):
        os.makedirs(dirname)


def chdir(dirname):
    print_cmd(['cd', dirname])
    os.chdir(dirname)


def get_config_vars(config):
    if platform.platform().startswith('Windows'):
        if config:
            raise RuntimeError('Single-configuration not supported on Windows')
        return {}
    else:
        if not config:
            raise RuntimeError('Configuration required on non-Windows platforms')
        return {
            'CMAKE_BUILD_TYPE': config
        }


def get_coverage_vars(is_coverage_enabled):
    if not is_coverage_enabled:
        return {}
    if not platform.platform().startswith('Linux'):
        raise RuntimeError('Coverage builds only supported on Linux')
    return {
        'CMAKE_CXX_FLAGS': '--coverage'
    }


def get_version_vars():
    # Get VERSION and VERSION_DESC
    git_desc = check_output('git describe --tags --always --long')
    version = re.match(r'[0-9]+\.[0-9]+\.[0-9]+', git_desc[1:])
    if version:
        version = version.group(0)
    else:
        raise RuntimeError('Failed to parse version X.Y.Z from `git describe` output')

    # Get COMMIT_DATE
    git_commit_date = check_output(r'git log -n1 --format=%ci')

    # Get BUILD_DATE
    build_date = datetime.datetime.utcnow().strftime(r'%Y-%m-%d %H:%M:%S')

    # Get BRANCH
    git_branch_output = check_output('git branch')
    git_branch = [line for line in git_branch_output.splitlines() if line.startswith('*')]
    if len(git_branch) != 1:
        raise RuntimeError('Failed to parse current branch')
    git_branch = git_branch[0][2:]

    # Return collected vars
    return {
        'sudoku_VERSION': version,
        'sudoku_VERSION_DESC': git_desc,
        'sudoku_COMMIT_DATE': git_commit_date,
        'sudoku_BRANCH': git_branch,
        'sudoku_BUILD_DATE': build_date
    }


def get_boost_vars():
    if platform.platform().startswith('Windows'):
        return {
            'BOOST_ROOT': r'C:\Libraries\boost_1_71_0',
            'Boost_USE_STATIC_LIBS': 'ON'
        }
    return {}


if __name__ == '__main__':
    try:
        sys.exit(int(main()))
    except RuntimeError as err:
        print('error: ' + str(err))
        sys.exit(1)