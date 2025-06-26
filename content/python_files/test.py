# %% [markdown]
#
# # Forecasting

# %%
import sklearn

sklearn.__version__

# %%
%pip install https://github.com/skrub-data/skrub@771f3870a23438211faefb0e54132781256acc0a

# %% [markdown]
# Install the development version of skrub to be able to use the
# skrub expressions.

# %%
%pip -q install https://pypi.anaconda.org/ogrisel/simple/skrub/0.6.dev0/skrub-0.6.dev0-py3-none-any.whl

# %%
import skrub

skrub.__version__
