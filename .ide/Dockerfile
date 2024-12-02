# .ide/Dockerfile

FROM ccmagruder/devcontainer-gemm:latest

RUN apt update -y && apt upgrade -y \
  && apt install -y --no-install-recommends \
    bat \
    curl \
    language-pack-en \
    less \
    python3 \
    python3-venv \
    python3-dev \
    python3-pip \
    tmux \
    zsh \
  # Configure shell
  && chsh -s /usr/bin/zsh \
  && ln -s /usr/bin/batcat /usr/local/bin/bat \
  && sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" \
  && sed -i -e 's/robbyrussell/agnoster/' /root/.zshrc \
  && echo "cd /root" >> /root/.zshrc \
  # Install neovim
  && curl -O -L https://github.com/neovim/neovim/releases/download/nightly/nvim-linux64.tar.gz \
  && tar xzvf nvim-linux64.tar.gz -C /usr/local/ --strip-components=1 \
  && rm nvim-linux64.tar.gz \
  # Install tmux plugin manager
  && git clone https://github.com/tmux-plugins/tpm ~/.tmux/plugins/tpm \
  && echo "if [ ! -f /root/.tmux/tpm_loaded ]; then" >> ~/.zshrc \
  && echo "    ~/.tmux/plugins/tpm/scripts/install_plugins.sh \\" >> ~/.zshrc \
  && echo "        && touch /root/.tmux/tpm_loaded" >> ~/.zshrc \
  && echo "fi" >> ~/.zshrc

