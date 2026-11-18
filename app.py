import json
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="CardioPredict-UACM · v2.1",
    page_icon="🫀",
    layout="wide"
)

UACM_LOGO_B64 = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAoHCAkIBgoJCAkMCwoMDxoRDw4ODx8WGBMaJSEnJiQhJCMpLjsyKSw4LCMkM0Y0OD0/QkNCKDFITUhATTtBQj//2wBDAQsMDA8NDx4RER4/KiQqPz8/Pz8/Pz8/Pz8/Pz8/Pz8/Pz8/Pz8/Pz8/Pz8/Pz8/Pz8/Pz8/Pz8/Pz8/Pz8/Pz//wgARCACqAQQDASIAAhEBAxEB/8QAGwAAAgIDAQAAAAAAAAAAAAAAAAUGBwEDBAL/xAAUAQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIQAxAAAAGZmISTcjkjAMGSEsyRgAEJJsQWcmQADQb8QtMWbmsXJNTRuMletSWgACgbldTEaAGKvtCrx7NIXNAxnBV75C+JmAcVUS1WJbThHWT4A0VfIIybZRKOghsXtrQQqd69hT+vZtLIaVbaBirZFFjlsmtrJHoBir7Qq8ezSFzQMZwVe+QviZ6tsVIVP4BY5qrqy65LV3xCXlXNlDgnYAABjOCn5vCJ8QV++rk92GgnhTlk1tZI9ACuLHj5GrFpmVk78RqOCyawm1joqqc1yevVr7yofNwaSq7Zp+fimO2lVxae+qpUSvVGIoWhur+wCn59AZ8SKEzcNGz34KcsmtrJHoAAEci1mYKsYWHkWsgIdiZAAAAQ7XNQOXqCCJ7TwVc8muTTtyFcy1yAAHn0FbTRqAABhWNRSDYV5GZycg2FmBocPONSP9wzF+ka5SOjJF3p1nKsHuODkHYp3neJugYibeMhb1m8AwifYEW9tkj+12CNXLwjLjvCPc8oyR7XJAjXuRAl6GYIuWT4Iz2OgR+XwKuCSgk5JLkUDcIqyb5AAMCMZdUe4iX8fnkHGYnITzza3Jp3oeoacifySTKHkJLrVdp38q7SSHMRenrqS6xxmObCU6U2B+JsjjKTsO8AFbQFXhwGrl7wSsugF+lsHDwvAVanQKsNgSsugFnhsChjuDj7AFXpmHBztwX+WQJWm4AAAA07ouSLzFt476It6JZo0RkmOlUqJcRzlJd5j3GSnak9DnjUaCUaeDnHntTzEl4eFISnqjIPO6LOzvAAAAAxkPOch5zkMY9B5zkPOch5zkPOchjz7Dyeg85yHk9B49ZDz6AAAAP/xAAtEAACAgEDAwMDBAIDAAAAAAACAwEEBQARExASFAYgNRUzNCQxMlAhIiNCYP/aAAgBAQABBQL+iu5Owm3ibjbXsu5Owm5irjbXsyGTcu3SyjitexrQSFjNTuWTuFocncHVfNTupoODR5e1B43Jk9/XI2/Er/WLWsc47FPpk/kfT/synyPp/rbdFetO5F+00H+RU6OaKVXLTLTa1Vtogwg7MwkbWKzax4RTpdpv3YmRnH2otVtEUAN6zNqzrDfG9Mn8j6f9mU+R9P8AXO2NzxKOa7l0cN3Bv7H9M8//AHrJmxYSoEK6NUDlrAVL037spKK2PtTVsxMTGbt6rIKw7WG+N6ZP5H0/7Mp8j6f6MOFrcyXOw6OGnl0c1JZytiGw5OsmXdkcCG9j3N+7ilC7FWElXfRyfBTIpM8bU8arrDfG9Mp8j6f/AH65P5H0/wBM4/sQiAJ0ZGnEfUaerEAL8E/cdZKO3IYA9n+5v3cH+BlqfkI1hqfK0v46w3xvTNB25DCuhVzoRQA2GctjAhtV1ef5FuFmUcLdcLdSs4iq6a9mJghzqNm1XTXsKYLl9GGKwS0HL037uD/A1exhFeUsUqL+OsN8b0y9WbFfVXMGAzmkbXck21CFG9qFQhGXfw0gGTNCoSjo5cNSwJWzCv5alhIvTarnWdUuNqkvNqmGZtW1q260WFmx5Gm/dwf4HUv46w3xvW7i1vlmLthrwLek4iwc1Kaqg6ydS1as4zHNTa9mTxzXWcdTt1bWnoXYXYwzImaFuNDj7ZarYWd1LBQaPF25PFoZXqdZ/b6Vc1jUmin/AOEqTMvtTMW5uTzRYB1UHgAsdAO8yZllxQV/L/4UuMzQ1K6tm0c1AYS6abPe2bcyePOGN1ynzWHQhVh0ITYe0Mg2zxjVOFvm3PKmyDFeaXa62Cgm92Eq1JOZYaJpaLldOSKlvki3dq/mf9pVLcVTmbLEdiFsWEVLnFwVGfqoWR46zdS2ow+0E8c5KrYXUTj5kmXjKE+G7gLkZi7ttT6lsxVeccLtsaNt9X8o1E2PPTxQolKs/mu+SBgFOL/B/pOb9X05v1fR9gEai52kJzLej3cMdDLtBDOVGpdta6NtLVZYYrXWdFhAMknaaUgv3dnl2ljKMkshcqmZsq/tlz7CCqctqq/zldSRcwTP0vuNq7VfhmzxssBZYukyn2IofgEM2ror4coswsBRNh1rShfkFi6ywGkrE7wo7bSpMsCacW3d91geEgaXcqk2X1OjKx86ashZis5Wlj2BNeCsjVdCUL4UWK3KcV3mQ14h0U3cJVN0FUa0prtBwU4hfivJaF8KG1yly6pDa8ZypUMguU729DTjwhVZ7/HiXzUnw315MwQc68WxAKWKlf0sMAmCwDEHLNK7ldpG0AOZ2jnXwiwSKLKZ1LQhvKHNDAlvMvsTZS+TaAFLQhsX6pE5y0CFuuYJsKfprQUOit1xAblcglgwZXK4M95KOb4MIseuSTomKdjnw201bYfSIH/SQeutbFHIALJN/d2iXLb4T3VzYuxcyKodYBLAv453arKTsJGttSi4jfc7rNqtcEK2/ap5C6rENC+h1dePpCYU/ftGttRER12jaYiem3TbpERHXtHUxvraNRER02jW0TraOm0b/wBL/8QAFBEBAAAAAAAAAAAAAAAAAAAAcP/aAAgBAwEBPwE7/8QAFBEBAAAAAAAAAAAAAAAAAAAAcP/aAAgBAgEBPwE7/8QAOxAAAgECAwUFBgQFBAMAAAAAAQIDABESITEEEBNBcSIyUWGBFCAjM0JSYnKRsQVQgqHBMFNgksLh8f/aAAgBAQAGPwL+RSRphwqfCpeLbs2tYe5JGmHCvlUvFt2bWsPcaOBgFXI5c6RZ2BRstPdxyMFWrbOnq1fNt0FfNv1FW2iP1WscbBl3EdjXwrhT2ue6R7lx327or6P+tLJJbEb6b5utT+nuTdan9N7yeAyq+pO5H56N13tI57K1ifTkvhVol6nkK+JMb/hFfCmz/EKwyrbz8aMisVi5/i3N1oEZEUG+sZNuLMbAUX+nRR5bk6nfN1qf09ybrU/pvWAcs2pSe6naNFh3ZO1RhOj6dd6QDQdo0sS/VQjjFgN5SRbqaCILKNzdaWb6SbUG+g5NQI0NezIfz0I0/wDm5Op3zdan9Pcm61P6bmdtFF6aRtWN6xHvSZ0SO9H2hSuuqm4pJF0Ybputqkf7V99utNG+jMaaN9RTxvmy/Los2ZNFmHxHGfluTqd83Wp/T3JutT+m5YRq+Z6UokbCl8zVhMP0r5w/SnETYkvkaeA8u0u6b816kTxX3261/WaxoPiJ/fdx3HYTTzNHcnU7yfvANYW0kFvXeWY2A1p5PuN6d/ubc8nLQdKuqMR0r5b/AKV8t/0q5RgOlJIORzoEaGlnGjZGklHKg8Zup3l3NlHOg8Zup3N1r+s7lMI7Ehz/AA0saCyrR3J1O/Eg7af33YZ1x/iGtZJJWC2CPw8aWOMXJpI10UUQO9J2RQVdSbCkjXRRvaNtGFqZG1U2rhnvR5elNG+hopIOh8a+GeydVOlfEjYHyzr4cTE+eVXkOXJRpVo/lfXfTc3Wv6z7h3J1PuF4zw3/ALGvl4vymvkPXxLRj9asgux1Y7ronw1Fh2q4k6gYRlnz93iQKDiHazoMyDAcm7W7BKtxV4GDDwOtfIb0r5LeuVX2h7D7VoJGuFRuY4Br91YJRZsV/d7g/wCwpY5BZhf/AILtVzpJ/itksdWP7VJFFC0jpUxZWGAEOnOtmUKQsi9k+FRxWuz/ANqbgwPIi5Fh/io5tUc2v4UZGidReyg6tWGSB4/PUU8maoGN7m9S8TZ5I0ZDZjUGGJpCVGQ6UYpI2jktex50wggeVVNiwraWXQvz6bvbcR4OPBb8Pj+tcQi+YoyEXtUSpG7LhOQPepLoxkfSMa0UkjeOWU3uxviqSKKFpHQ507m6YO+G5VxPZpOD93/qonsXWQ2GGsM0Dox7g1xVwpYmie1xfO9ELssjKOeVLIndO+bjZRynEr8qhMNzHFcl+VbX+YftX8R6f+NRhe+qKy9RT7UQRcYE8vH+9cKfaZoXTlyPTKtmCh8LTg9vWvjqSnkNKCQztNDhzxfT602BblZi2HxzqRYru5U9m2nWtlWR2jgMebLlnUfBZ2TAc2JI9K9n2g4HQnl3q2lipW8mh6Vw4+/KcArg+0jh2tbhCnjIvLC1j52NYYSWJIvlpnWzyyZJZheotqzMJTDe2nnUCwdoI+Nn5Ctr/OP2r+IImpIt+lWz4lvlWzvWwI+okzrY+rftWy9Hp/appVmxdwEj9KXXvNr1/kvAt9GK+8wW0TFfeAbs7d1V1NATQSRBsgTmKdShAGjeO9Mr4nC72bwF6SS1sQvbcsNu8uK++OF9X0NF3NlGtLKoIB8adeGQq/UeZ3FlQufAe/MJGYRxHCFU2pruXUQXF9da4kx2jitmCgay9KRpQQ/O4pyf9j/NNJENrPMSXqN21ZQTW0YtQq4em7+Ido5ILeWVBr58HX0rY4MZUSR4nYHM5VBw3bBxlurG9EWnkK6iM5CtosWuj4U4mov407LLJxMPeLa1B+QVLG7MI4rdlTa5NIMZZOGbXzIrize0Y20wBrL0ocW+IZZi16WN9DEf3pYdoHYh75+88qhCkgvJhuBmMzSNsw2i+LtBg1iK4oOJJMsBOh8alJkLSHMtf9q9nxFY1TE2E2xVLLG7kBckY3F6DGaTjEd/FSO3e57zLs8vDZu8CLg0ZnlxsVwnKsGzzhY+QK3w0FLFrczTSk5NHgtXB9o+Fa3d7VqSO98ItelkRzHKujChxto7I5IuG9TuTcS2Fq4HtPwbW7udQqr4ZIR2XtSNNPcowYALYU7wShRJmQy3qZJHMglNzXCk2m8drd3M0kd74Ra9caCThvazXFwaE8kuN8NjlRGzTBUJvhZb26UFLlz4mlmvouG24QM2huGHI0vE2i6r9q2vTyynHcYVFtBT7PxOye7loKWWJ+HKuV7XuKf2mXiYhbCBYVwl2r4Wnd7VutLGgsq/yZowwxrqKLK1wNa4qsCnjWGOVSfClVmAL93zq5oS4xgPOmVTcrrSWkHb7vnQjxDGRe1cLEMdr2oxhhjGopmxjChsT4VaKQMRSq7AF8h50IywxtoKCiZbmsUrBRTMsq4V1NHhOGtWKRgo03BmlABNqxCUYb2vSoT2m0FcNplDf6G0yw/NjK2H3C2lcONTxJ5GAHlzraIXj4SuhZFvflnUMMXb2iy4bfSaleOLGF7MbYrWI51xBzWo2MwMfZ7OHz8a2njtgxEML88q2KNrqSJCPLmKil2hryMrFzXtvAPfx4sX0eFTywH4iBCnn5VxXQ8MbSWdfCtn9mOJkN2YchWyxnK5b9q2d52xTPiv+lQqdrjA/wBvDnWznGE+L3jypxLtQexHaRO7UiFklAHzUFvSuGsXESIZ9q2ZpV2lsDqcDX8RWyvjEd5HIZhSK0yzfGUHCLVBA2eAPgfxFqMMwtIMmQjMmolk7wX/AEshvtWY97Ib9BWdaVlu0rStN17Z/wAm/8QAKhABAAICAQMDBAIDAQEAAAAAAQARITFBEFFhcYHwIJGhscHhUNHxMGD/2gAIAQEAAT8h/wAFoagvxiPsQbv6MyUovxn2kg3f0cSIQ3yjdQ4Snj6Woxyy0DXzxELv4ART8QspAV88QqocnQHYCJSU9IrPb6BwTiT99GyhC1K56/N8E3+fP0fhP0T5ffqp94nd4iUrVb5ZnkEluuvsOtHILYraj6QmewNuIb8+g/c3p4f6JfHbQyehhxDjtXbp+T/cctRYnERNaXz0KgO1eCLOweg6PhO/X5vgm/z5+j8J+ifL79bj4++4+eZXC/6T8ysHA9efnmXvx36H9dVU8Hu8TZIt9jln7zL89S7cRhzy0B0/J/uHHL34SIm9L4jtiFicz8qR+CbPHK9nL0+E79fm+Cb/AD5+j8J+ifL79Eppi6dwVp1k+nHzzKbX9o5iQ0RJydD06Ix4/FiElsA93+vr/J/uB/ZHpqb4nD3O80AHu8e0SBXa92c4U8XB0+E79fi+CfJ9fo+b4J8vv0vHk9l/f6gnkyuCBAwwFv8AUUKa/V/qHQ0rtL15+w56ON5/JmAtsk9n+/r/ACf7m34anBTs8OTpyCMX4Yn4HT4Tv1WrwJ+v4loqoP06lQHaeCeWmFa7R6EWi3Uz5v7LU88AEz/oZ/0MfCHK54xp3OY4dhYximD3zX4/Uy83ydzkld02PUeZLVB78zp+T/c2/DXSgBnawuZXCFE/A6fCd+oi9nDs5JkezBbycle7vB5J7UH8yoPAtvqZaQf28zijHrLb19o5h/WYPM4sh69dugULqmKZRy/6IHeHfbzP05AdyLmzyBAns6IBetoJrL9gIBja9B/fT8n+5t+Gvo/A6fCd/otbO8QRxT3Bgz/BBnlbbfYjn1QD0qgchBfdggxwT9LxOAGZTcGAa6NA8Xc9Iq9G3/SO0+yuO0F7ISDe4feCy0A6HypTpKFLFBv6DbDt0oD0iwN8/wDwokiUFusIIIFoHcMS2WCBVbtjIVAwMQgthG9LBgmbXXA2sViupBabo7loUX1Hf0qY0+hNuqOIkaVYk+4cwXm/wu6lBAFRymLDJEIq8OPJZVCtO07iS3AJAL7F7lu1HSkw6fwofGSFRoo8tRkQQUeWv5mqKEhp+oB+E4+XjEWWCq9ATt2jqjICBVbtgt202GT3fa3XflUGFhIu8YiCW8FeljmXOmmgD1IgpclF+heYnNji+pn0ihdqpHtOGTtSUoCHhmp17E4xWHsVkqFMDsHw9pl+CVcF75Ipa2DyXcyxOlrPqxqJkyLyvx/SWWeOQvUqOGz3hl7KiDULjoKF4ly5NwY2YwRQLWC3Z3mUy1FOkI/gje32J/HTH3jThQmwDZ7R4xOixpuIhW2uBaqIEVopzbKdonNUAwBq+7Dwf8s/I1zCGeNbL6VQPuM7WLUN4pTkr/kSqIoxzOKG5YsBwbbdd/VR1CtdKm+lH01N9ADXWi75+ii7+n1Hn+arqF3IZ/NV1rLiBXBVoKIs81qUZXs16epJe/NVfPXFl3UlE4PJrpz/AEyarrYUD4PSUrDalvI6Nt1MPVcXgD+eg2Brcw19VhUiWzVq1BNJW7BwvmF2/ZKCmJXTFYF07qCKKAK++XbORuvtevaflGoqVVr1/u/PSlYG5d3Uys3ZZuQy1gABi/MF9kKpu9l6iqfgVPzmZZhMgxUr2XMe6lUmObNVPgO0BC5s0l21HyaZewYvtHKtHWN4wxFc3lNymmp3TH5GmZRDXwfxOY4rWeBso7y9rUPlbu+Zz1Vw3A8d5WVk3JT2ErlviCWgvtOCggndnM2D5V4fTVSuO1lOUa6lGCDEnT6x82faG+Ox4hylu3+huIsIp3svN2tfPeYdUpr2l3/E43PBVygYqFeOyckyTrf5wygkUrqioJcgDj7XLwRA/CnHZnCesI/mKLjTb3My+UISnUXuxypdlnG54KuZ6XQTxZHm2NQ9jiCEu2O72gqpvezvr8Hd6HWqx1NliR1Gi6v+tgWCsOHOe8Vt7Ns3XXmVsRTE7CQfAa4T07wsgUouLs/2lCIUH+Gckp75cMApF2TcwJla+Nwu16tlibyoMAiUGWIpwVccwNK6PZmZm6ehuJXFHOp/KV1FmBvlCD5eFNCNx5twG4nlqXlC1gXlQjAiUGcsGLTQveKNDfahXdoNkDLLk9+jpKB8m4kfBfk6hey3zVuZqLSXr/w3LZwbKi/KEaS1/hNDp9pVBZ94o4FQ5CrV4j/sGi0to7zNJr7OzWSA8SrdIA/QJOyLc3LSrs/UIWPmgAoPQlNLkauFYekyVarwUbXhiHiSd08nhj+FSOKdLEZRmGx0ZQuF5oDAPnMCwlVu9q5rgDBZg5qBZ1422LIKHLsvLzA5zoQYWN9iWak4HqekHQW3UB5qXNdnTeIvLkvkwvyQcgn+2e5aTUEeProta3Ml0XEOwmgD0IAFBUACgnAFdpQUNdyUXdZIh2DKKriANASg0TSh6Eosa1KLutT/AIUDQPrAtAvxAKAHiAGjcWuw3vEdwNeIHofaUXdal2B5V/hv/9oADAMBAAIAAwAAABAwAQQAQQARiwwwAQwCihSgCRwBwCBRAAQACiBSBBQBTgABQSBQAACgRwQQigCARRTDQAADTiCAADBCShAAAAAAgggggjASQQBgwgAACShDiDgRhCSzhCDDABzhCSwQCRCyQTARiABCDCAACBCDBAABBDAAATzDwTQRzzRzQBQAAACABCCCAACCACAAAD//xAAUEQEAAAAAAAAAAAAAAAAAAABw/9oACAEDAQE/EDv/xAAUEQEAAAAAAAAAAAAAAAAAAABw/9oACAECAQE/EDv/xAAoEAEAAQMDBAMAAwEBAQAAAAABEQAhMUFRYRBxgZEgofCxweFQ0TD/2gAIAQEAAT8Q/wCFbti7MQbs81o5/NCb8HwsazdmIN2ea/23ITfg+BEUEdaXJpMeGpyxsSTKQ3g8/HL+V8uwZXgpg8bF5eQtvL4pzCbOPqiLI7KfqpDjszicrb+GsP8AV8OyZHh6Wl6J0im9QwbiaMq+pjtGvwST9nCdUbB9oV+n+qsqCsFkFvHRxX1VX0Ou4+Pp4m5EmrYe0pbUaZUf/aR4i95siUR4nsrD7s+et6NJq7BytilYoQn/AFO7rT2U7vyH+i9GC4yCDzKalkYwNHzh6a/SCiL9s1YALJOgBxmdMa9Pzd1ERItCi4lOIIg6bzhz7NOIng44CVrMiC0seXL36fo7+jivqqvoddx8fT4v+IdRA8EvhW7mzCjb9HsNW6zYcErfd6We4GuC/uXo6phICaqQPYl8lWZYrCYr+AFo5RoN1qnVd+pjFvoOiOiblCYPpx/7zTX5u6hCncGmQe427NOIoA67zkyeTWiYsWkDcSsW2c8/2vjmrAUqyw3T9mCnNq/R39HFfVVfQ67j4enn2IeAmnVv0bTg7BB4q6a72TE+r0vhixlBb6v4K4AOAjJWIlk3anhk8dFHWy4Af0ouJ4CzfXyNfm7qnU7bqIJyMJRv6ALaQ4SoPs/IUkxLhM9pNqbS191GVq3FfWYP9XntTmv0d/RxVk+ypEM3hx1uKYubCoMOLdG2PGBwljzD2oXAb7dcaxbzQMLAQgYKP0UQjClBNBpEF3C97THis47Auqx8MPl6C3Cg7AH80lgcpZPka/N3V+htTMpAhfP3dTmTWsVknaiw69su8c19v/FOa/R39VWhi3gl90elnJYIM+1zunVzwcUBK0jZEcOgtj1FDnB3QYn2vqgRABKulIUlwdrHtnzRVqwcfIV+L/qvxf8AVZO8GB5SpioKDVsPS0QUw2ESRp7V7Gn9ygQZZ/PbyC0dyKH8OyYTrPr0sB/vFAdWwyOomibU1+bur9Dbol1khuTeElObbVEYE/7eVu96+/8A4pzX6O/rfecpdcHNhOzvQjRQZEslCJGACHBsubVNdNoXuiMTstsMbnYt3pSkIbDVbAXawxYuq1XlZfNXYRYbiL/VvJUL+g1TBWApz3avll89SXnjiTPcb+KhFAeRipybAS3Vd+LngqX5Ghl6Dkb1DkS5nC/8NKLmyb/ljR5KsV68vYUatMG38QVoVds+gTV5b1OThmA0R+u+trlfm7q/Q2+D7/8AinNfo7/gfK7jJ90MPJ6p6AOlHww/VAAm5J/dPm7D8P8AYlNSBruPg4PvoPPLKS8C2lt4KWoZJNVps6E+Y+JgwEGCbTfcj1RlSp3drE6IPvooJ3TZd1kanxqzxcTl9U7fTUD7FobLdYh7SgtS6srhdjxNY+tP7d3l6KXiNtV3pxZrG1CLnb4Jmig9Uzfu90pIoEQpLnHyg/5KgS2KGSTHURUEYz8xGYcf/JxReciiDAbUVXtgBNua0RiKb4MmgzAZYaVjjs8agjqYSr7xxYFJuqa6pUWC0ZAp4WDvRdkl+ARMeKEYPi2SUPJJmgx7vAji4yd6uQlnCS0lrsRvV3gIkM0XcuDmlBRKiSSl1Ql1b0eJIowMoAKSIPlCWJCGHNMhZD+zKJnFXBvAiIkRwiInFNimLzu/Vd2o+NkoNoN+9Jd7uCwjNJuCJ8RdCntvagB1qwEyZgamYoZfk+jgyGIFt6PHQH0ETQZgMsNPTDXAksxki8mawCQz+74L9qKhCVJJgaziOal66UzkIKguJGwa1McMONvFiTUqNRruJaVGCSUEIjCJoiJ46u4nuQASBlYSggoNOTTQlurSDqBiYzbSUSwrZ4VNtDlcH2RHmiHojgKV5TRhUQOSCX0EZyszUa8GqFMAsOfNODhmB5NYCZpTTBUwSJEmb3XIpKgTC7aG9v4qV34AGVwQLnN4tNXJLEBie5idppgGGNUpFyxrFqKE9LDIcG6cZmonqGMDApopCmk0vaHUO97EjPalv45zGdXOZvQ4+eYUbkmeb0Dhltp5mWvBGZaVEtaRoFMTDVued4jISItMWpsy5YoCYvJg2phVEwMZoH5I+2EPE8xHmjgIy6YjFvriKztIFx+iTHilQJCZjF1MmUEzGKWgSycyDr5F7zNEBfge9z1QEJI6NABAQG3xURQUxJjqAgB26AFQBc80gIQR0aACAilEUFMcfEBMATdgpAQgmz0AgAbB10A5Rf4WECTDHxhCXsTjfc9ZnzQoi7HiZ6z0qyfmYNA1W1BPjLAYBNk80UCZouEsM2xfqqWHhcOO6NuoOo3BEwLH1RvJBUriYnXoUheTQER56yqeXi+YFrK4ovSbNArHFUoQro7UGQYLRsoJQOybdCTeSwuhabc+KShSFMbfIHOFi4TguYL0aTBhua8rhhbwxpRxaQKdYAkFszN6mkMaNIXMSA+aMAtDABcqyTWxMlEYn+KoZrIEEhP3RvttcYsf5UglygAhZnCWXZe9qWx0WlqJnMzrRFMIC0jIU3c1rbSJCMlWZJhHih2vnq3JYJDu0A4BKB5GS+tqKU+pSUHbl8FrbV+HsoB9y4QJcgWCh9bl5uSXcInFyn3noRECGSBdZvQBT9QqFWJI8zRRWMGEgORhpoJLoO7OyFy09YUphIswEO9MxtZrQkYBkSKR3wgEKZOHoMZoAhq8Yq6wWA071GDcEsgFwRLGaFOaqTUAZAuSTvUgUETFJsXLpGKG8CsWSScMT566AYQeISIC0jVjRs0YJAYAI8laZwRDkMpEtewjFDRogL7sVpGFa0lbuYxUEm1RLECtYtJKNq9SagRMaUaDoVJZdbgpKc3OAyXFiTBFYR/0i7vN5mkurLAtwOeDFiUtaoIcBMIJFkC5NFrjb0UskqrEra8F6Mo2HIYnBEhhkqMeNwsCbWmSSC1jSmFIlFEQS8YmAXe9epNQImNKjuGKDtwXNEaMDhExqIE2EO6rK0Y3heJMkLSrDIUNY8KVZm1uxtWC5v1AzM8Y6XcMws5i5Mc04GVlWYGW24BNMjVAHYCWXU+MVL2mLcQNmwiBtbtSWFYhRldkm+bVexxzdzkrymoqVADjQlGLWTUcAZ02551/4xyUQNkSKURqe2EQGdqgjSZICXdaGsRGiHYGJ8UsprV0ZD2UDASjoFFyElmUoPa0XEJMsSD4pIMXuk3AOzagHFS7hlodWnN1YmjUnmXGHtUNzQuCK6k0PzJUkN4YY5plAVY4D2UuwLOoL9Pqi5vNkMBjerxk8wwWLcDQ4lSkC4mTWKTLNmpJiRuUDBIeysHRERPmFYGNKuBIABiXNaDqzTMUw7DS0QkbVsuB7tCIIyPzTu4phA86LOiFWWlylMi2IsnminY5AoBY2h3ovNKTjCWAM1CXmGTG1BCTQijQSbt6PAiUgCESRLRdo8UHucwxGHkiRFZ54ssO4LopMAfW02BJ5WpIl1FlysWve9FrPpoiAHmKlhtySB3BLlMCpnzDzGAKKfKPAphyIPigeBX2uE7SvKoAIkO3vVmWcxrTEyfz7tRWefNDcgmSRmay8xZJQ0UF7YozLTMcKbFuWwtSyHQUBlFiYI63isnjhpQlEyWO40JhKYDRIF5vV2fQaaeSQ+KTYH1Qtwi5JD22qLZlE2DyEHj5gAA5MZoAAhLMb5pMUJMKTE5qUuOwVGkNgilgAcgVaWXELeqVuFJAxSgQcDFyk5tiXJs5rQiyIi0UnNsCxFjFEyAllgy0QhKygFKIC4KYpYILgpigkQkwhoKBS8CaLQrKATWomoEFKKAqWDLQ4OUhDPeiAIwEMUJAyZgGaUCC4MYpFSiwhJ5/43//2Q=="
CIFUENTES_LOGO_B64 = "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"

DEFAULTS = {
    "imc": 25.0,
    "cintura": 85.0,
    "cuello": 34.0,
    "grasa_bia": 35.0,
    "grasa_visceral": 1.9,
    "masa_muscular": 19.5,
    "angulo_fase": 5.6,
    "fuerza": 15.0,
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)
st.session_state.setdefault("resultado", None)

def limpiar_campos():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    st.session_state["resultado"] = None

@st.cache_resource
def cargar_modelo():
    return joblib.load("cardiopredict_v2_1_modelo.joblib")

@st.cache_data
def cargar_metricas():
    with open("cardiopredict_v2_1_metricas.json", encoding="utf-8") as f:
        return json.load(f)

modelo = cargar_modelo()
metricas = cargar_metricas()

st.markdown(
    """
    <style>
    :root{
        --navy:#0b2d56;
        --navy2:#173f73;
        --maroon:#8f1230;
        --maroon2:#a8173b;
        --ink:#14213d;
        --muted:#667085;
        --line:#d9e1ea;
        --soft:#f5f8fb;
    }
    .block-container{
        max-width:1240px;
        padding-top:1.1rem;
        padding-bottom:2rem;
    }
    header[data-testid="stHeader"]{background:transparent;}
    .brand-wrap{
        display:grid;
        grid-template-columns:minmax(150px,0.9fr) minmax(340px,2.5fr) minmax(150px,0.9fr);
        gap:24px;
        align-items:center;
        padding:18px 24px 16px;
        border:1px solid #e5e9ef;
        border-radius:24px;
        background:white;
        box-shadow:0 10px 30px rgba(11,45,86,.08);
        margin-bottom:12px;
    }
    .brand-logo{display:flex;align-items:center;justify-content:center;min-width:0;}
    .brand-logo img{max-width:100%;height:auto;object-fit:contain;}
    .brand-logo.uacm img{width:230px;max-height:122px;}
    .brand-logo.lab img{width:170px;max-height:150px;border-radius:8px;}
    .brand-center{text-align:center;min-width:0;padding:0 8px;}
    .brand-title{
        font-size:clamp(2rem,4vw,3.25rem);
        line-height:1.02;
        font-weight:850;
        letter-spacing:-.035em;
        color:var(--navy);
        margin:0;
    }
    .brand-title span{color:var(--maroon);}
    .brand-sub{font-size:1.12rem;color:#253754;margin:8px 0 10px;}
    .version-pill{
        display:inline-block;
        background:linear-gradient(90deg,var(--maroon),#b21d44);
        color:white;
        font-weight:750;
        border-radius:999px;
        padding:6px 22px;
        letter-spacing:.02em;
    }
    .wave{
        height:6px;
        border-radius:99px;
        background:linear-gradient(90deg,var(--maroon),#d997a7 32%,var(--navy) 74%,#567ba9);
        margin-bottom:20px;
    }
    .intro{
        border-radius:24px;
        padding:24px 26px;
        background:
          radial-gradient(circle at 92% 20%,rgba(143,18,48,.13),transparent 28%),
          radial-gradient(circle at 76% 80%,rgba(11,45,86,.14),transparent 34%),
          linear-gradient(135deg,#f8fbff,#eef4fa);
        border:1px solid #dce5ee;
        margin-bottom:18px;
    }
    .intro h2{margin:0;color:var(--navy);font-size:1.8rem;}
    .intro p{margin:.45rem 0 0;color:#42526b;font-size:1.03rem;}
    .info-card{
        border:1px solid #dfe6ee;
        border-radius:22px;
        background:white;
        padding:22px;
        box-shadow:0 8px 24px rgba(11,45,86,.06);
        margin-bottom:18px;
    }
    .heart-panel{
        min-height:300px;
        display:flex;
        flex-direction:column;
        justify-content:center;
        align-items:center;
        text-align:center;
        border-radius:22px;
        background:
          radial-gradient(circle at 40% 28%,rgba(255,255,255,.22),transparent 19%),
          linear-gradient(145deg,#0b2d56,#164e85 54%,#8f1230);
        color:white;
        overflow:hidden;
        position:relative;
        margin-bottom:18px;
    }
    .heart-panel:before,.heart-panel:after{
        content:"";
        position:absolute;
        border-radius:50%;
        border:2px solid rgba(255,255,255,.16);
    }
    .heart-panel:before{width:250px;height:250px;right:-70px;top:-65px;}
    .heart-panel:after{width:180px;height:180px;left:-65px;bottom:-60px;}
    .heart-icon{font-size:5rem;line-height:1;margin-bottom:.4rem;}
    .heart-panel h3{font-size:1.5rem;margin:0;z-index:1;}
    .heart-panel p{max-width:280px;color:#edf5ff;z-index:1;margin:.6rem 1.2rem 0;}
    .model-stat{
        display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;margin-top:16px;
    }
    .stat-box{padding:12px;border-radius:14px;background:#f4f7fb;border:1px solid #e2e8f0;text-align:center;}
    .stat-box b{display:block;color:var(--navy);font-size:1.2rem;}
    div[data-testid="stForm"]{
        border:1px solid #dbe4ee !important;
        border-radius:24px !important;
        padding:22px 24px 18px !important;
        background:white;
        box-shadow:0 10px 28px rgba(11,45,86,.07);
    }
    div[data-testid="stForm"] h3{color:var(--navy);}
    div[data-baseweb="input"]{
        border-radius:12px !important;
    }
    .stButton>button, div[data-testid="stFormSubmitButton"]>button{
        border-radius:12px !important;
        min-height:48px;
        font-weight:750 !important;
    }
    div[data-testid="stFormSubmitButton"] button[kind="primary"]{
        background:linear-gradient(90deg,var(--maroon),var(--maroon2)) !important;
        border-color:var(--maroon) !important;
        color:white !important;
    }
    .result-card{
        border:1px solid #dbe4ee;
        border-radius:22px;
        background:linear-gradient(135deg,#f8fbff,#edf5ff);
        padding:20px 22px;
        margin-top:16px;
    }
    .result-head{color:var(--navy);font-size:1.35rem;font-weight:800;margin-bottom:8px;}
    .result-grid{display:grid;grid-template-columns:1fr 145px;gap:20px;align-items:center;}
    .result-value{font-size:3.4rem;font-weight:900;letter-spacing:-.04em;color:var(--navy);}
    .result-label{font-size:1.03rem;color:#344054;margin-bottom:6px;}
    .donut{
        width:125px;height:125px;border-radius:50%;
        display:grid;place-items:center;margin:auto;
        background:conic-gradient(var(--maroon) 0 var(--pct),#dfe8f1 var(--pct) 100%);
        position:relative;
    }
    .donut:after{
        content:"";width:88px;height:88px;border-radius:50%;background:white;position:absolute;
    }
    .donut span{position:relative;z-index:1;font-size:1.9rem;font-weight:850;color:var(--navy);}
    .classification{
        margin-top:14px;padding:12px 14px;border-radius:14px;
        background:white;border-left:5px solid var(--maroon);color:#344054;
    }
    .placeholder{
        border:1px dashed #b8c6d6;border-radius:20px;padding:22px;
        background:#f8fbff;color:#667085;margin-top:16px;
    }
    .disclaimer{
        margin-top:20px;padding:14px 18px;border-radius:15px;
        background:#f1f4f7;color:#475467;text-align:center;border:1px solid #e1e6eb;
    }
    @media (max-width:900px){
        .brand-wrap{grid-template-columns:1fr;gap:10px;}
        .brand-logo.uacm img{width:190px;}
        .brand-logo.lab img{width:135px;}
        .result-grid{grid-template-columns:1fr;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="brand-wrap">
      <div class="brand-logo uacm">
        <img src="data:image/jpeg;base64,{UACM_LOGO_B64}" alt="Logo UACM">
      </div>
      <div class="brand-center">
        <div class="brand-title">CardioPredict-<span>UACM</span></div>
        <div class="brand-sub">Modelo exploratorio de carga aterogénica</div>
        <div class="version-pill">Versión 2.1 · n = {metricas['n_total_analyzed']}</div>
      </div>
      <div class="brand-logo lab">
        <img src="data:image/jpeg;base64,{CIFUENTES_LOGO_B64}" alt="Logo Cifuentes Lab">
      </div>
    </div>
    <div class="wave"></div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="intro">
      <h2>Estimación exploratoria de carga aterogénica</h2>
      <p>Ingresa los datos antropométricos, de bioimpedancia y fuerza para estimar la probabilidad de presentar ≥2 alteraciones aterogénicas.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([0.82, 2.18], gap="large")

with left:
    st.markdown(
        """
        <div class="heart-panel">
          <div class="heart-icon">🫀</div>
          <h3>Perfil cardiometabólico</h3>
          <p>Herramienta académica para explorar patrones asociados con carga aterogénica en la muestra estudiada.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    m = metricas["metrics_oof_nested"]
    st.markdown(
        f"""
        <div class="info-card">
          <h3 style="margin-top:0;color:#0b2d56;">Acerca del modelo</h3>
          <p style="color:#475467;">Regresión logística penalizada L2. El desenlace corresponde a la presencia de <b>2 o más alteraciones</b> entre siete marcadores aterogénicos.</p>
          <div class="model-stat">
            <div class="stat-box"><b>{metricas['n_total_analyzed']}</b><span>participantes</span></div>
            <div class="stat-box"><b>{m['auc']:.3f}</b><span>AUC-ROC</span></div>
            <div class="stat-box"><b>{m['sensitivity']*100:.1f}%</b><span>sensibilidad</span></div>
            <div class="stat-box"><b>{m['specificity']*100:.1f}%</b><span>especificidad</span></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with right:
    st.markdown("### 🧮 Calculadora de riesgo aterogénico")
    st.caption("Las ocho variables corresponden al modelo v2.1. Edad y respuesta a bicicleta no se incluyen porque no mejoraron el desempeño interno.")

    with st.form("formulario_v2_1", clear_on_submit=False):
        r1 = st.columns(4)
        with r1[0]:
            st.number_input("IMC (kg/m²)", min_value=10.0, max_value=70.0, step=0.1, key="imc")
        with r1[1]:
            st.number_input("Circunferencia de cintura (cm)", min_value=40.0, max_value=180.0, step=0.1, key="cintura")
        with r1[2]:
            st.number_input("Circunferencia de cuello (cm)", min_value=20.0, max_value=60.0, step=0.1, key="cuello")
        with r1[3]:
            st.number_input("Grasa corporal por BIA (%)", min_value=1.0, max_value=70.0, step=0.1, key="grasa_bia")

        r2 = st.columns(4)
        with r2[0]:
            st.number_input("Grasa visceral (L)", min_value=0.1, max_value=10.0, step=0.1, key="grasa_visceral")
        with r2[1]:
            st.number_input("Masa musculoesquelética (kg)", min_value=5.0, max_value=60.0, step=0.1, key="masa_muscular")
        with r2[2]:
            st.number_input("Ángulo de fase (°)", min_value=2.0, max_value=10.0, step=0.1, key="angulo_fase")
        with r2[3]:
            st.number_input("Fuerza de agarre ajustada", min_value=1.0, max_value=60.0, step=0.1, key="fuerza")

        b1, b2 = st.columns([2, 1])
        with b1:
            calcular = st.form_submit_button("🧮  Calcular riesgo", type="primary", use_container_width=True)
        with b2:
            limpiar = st.form_submit_button("↻  Limpiar", use_container_width=True, on_click=limpiar_campos)

    if calcular:
        datos = np.array([[
            st.session_state.imc,
            st.session_state.cintura,
            st.session_state.cuello,
            st.session_state.grasa_bia,
            st.session_state.grasa_visceral,
            st.session_state.masa_muscular,
            st.session_state.angulo_fase,
            st.session_state.fuerza,
        ]], dtype=float)

        p = float(modelo.predict_proba(datos)[0, 1])
        st.session_state["resultado"] = {
            "p": p,
            "inputs": {
                "IMC_kg_m2": st.session_state.imc,
                "cintura_cm": st.session_state.cintura,
                "cuello_cm": st.session_state.cuello,
                "grasa_corporal_BIA_pct": st.session_state.grasa_bia,
                "grasa_visceral_L": st.session_state.grasa_visceral,
                "masa_musculoesqueletica_kg": st.session_state.masa_muscular,
                "angulo_fase_grados": st.session_state.angulo_fase,
                "fuerza_agarre_ajustada": st.session_state.fuerza,
            },
        }

    resultado = st.session_state.get("resultado")
    if resultado:
        p = resultado["p"]
        pct = p * 100
        clasificacion = (
            "Patrón compatible con ≥2 alteraciones aterogénicas."
            if p >= 0.50
            else "Patrón compatible con 0–1 alteración aterogénica."
        )
        st.markdown(
            f"""
            <div class="result-card">
              <div class="result-head">📊 Resultado del modelo</div>
              <div class="result-grid">
                <div>
                  <div class="result-label">Probabilidad estimada de carga aterogénica elevada:</div>
                  <div class="result-value">{pct:.1f}%</div>
                  <div class="classification"><b>Clasificación exploratoria:</b> {clasificacion}</div>
                </div>
                <div class="donut" style="--pct:{pct:.1f}%"><span>{pct:.0f}%</span></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        report = pd.DataFrame([{
            "fecha_calculo": datetime.now().strftime("%Y-%m-%d %H:%M"),
            **resultado["inputs"],
            "probabilidad_carga_aterogenica_ge2": round(p, 5),
            "version_modelo": "CardioPredict-v2.1-depurado",
            "uso": "investigacion_no_diagnostico",
        }])
        st.download_button(
            "Descargar reporte CSV",
            data=report.to_csv(index=False).encode("utf-8-sig"),
            file_name="cardiopredict_v2_1_reporte.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.markdown(
            """
            <div class="placeholder">
              <b>📊 Resultado</b><br>
              El resultado se mostrará aquí después de calcular.
            </div>
            """,
            unsafe_allow_html=True,
        )

with st.expander("Información metodológica y límites"):
    ci = metricas["bootstrap_95ci"]
    m = metricas["metrics_oof_nested"]
    st.write(f"**Muestra analizada:** {metricas['n_total_analyzed']} participantes.")
    st.write(f"**Distribución del desenlace:** {metricas['target_distribution']['positive']} con ≥2 alteraciones y {metricas['target_distribution']['negative']} con 0–1.")
    st.write(f"**AUC-ROC:** {m['auc']:.3f} (IC95% bootstrap {ci['auc'][0]:.3f}–{ci['auc'][1]:.3f}).")
    st.write(f"**Sensibilidad:** {m['sensitivity']*100:.1f}% · **Especificidad:** {m['specificity']*100:.1f}% · **Brier:** {m['brier']:.3f}.")
    st.write("Validación cruzada anidada estratificada: 5 pliegues externos × 10 repeticiones. El modelo es exploratorio y requiere validación externa.")

st.markdown(
    """
    <div class="disclaimer">
      🧪 <b>Uso académico y de investigación.</b> No constituye diagnóstico, no estima riesgo cardiovascular absoluto y no sustituye la evaluación clínica ni el juicio profesional.
    </div>
    """,
    unsafe_allow_html=True,
)
