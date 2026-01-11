# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
    . /etc/bashrc
fi

# Initialize conda
if [ -f "$HOME/.conda/envs/ssl/etc/profile.d/conda.sh" ]; then
    . "$HOME/.conda/envs/ssl/etc/profile.d/conda.sh"
elif [ -f "$HOME/.conda/pkgs/conda-4.11.0-py39hf3d152e_2/etc/profile.d/conda.sh" ]; then
    . "$HOME/.conda/pkgs/conda-4.11.0-py39hf3d152e_2/etc/profile.d/conda.sh"
elif [ -f "/opt/software/Anaconda3/etc/profile.d/conda.sh" ]; then
    . "/opt/software/Anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    . "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    . "$HOME/miniconda3/etc/profile.d/conda.sh"
fi

# Uncomment the following line if you don't like systemctl's auto-paging feature:
# export SYSTEMD_PAGER=

