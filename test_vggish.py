from vggish_input import wavfile_to_examples
from vggish import VGGish

vgg = VGGish(postprocess=True)

log_mel = wavfile_to_examples("data/audio/samples/ACMZ/ACMZ_AC_1.wav")
print(log_mel.shape)

pred = vgg(log_mel)
print(pred)