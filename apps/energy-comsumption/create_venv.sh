update_requirements=false
OPTIND=1
while getopts "u" option; do
	case "$option" in
		u) update_requirements=true ;;
		?) return 1 2>/dev/null || exit 1 ;;
	esac
done

if "$update_requirements"; then
	uv pip compile pyproject.toml -o requirements.txt
fi

uv venv .energy-comsumption-venv
source .energy-comsumption-venv/bin/activate
uv pip install -r requirements.txt