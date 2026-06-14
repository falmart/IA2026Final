"""
Dataset de textos con segmentación anotada manualmente.

Cada entrada contiene:
  - 'id':       identificador único
  - 'texto':    texto completo
  - 'segmentos': lista de segmentos correctos (ground truth)
  - 'temas':    etiquetas temáticas de cada segmento

El dataset fue generado sintéticamente combinando párrafos sobre
temas claramente distintos, garantizando alta coherencia interna
por segmento y baja coherencia entre segmentos adyacentes.
"""

DATASET = [
    {
        "id": "D01",
        "temas": ["cambio climático", "inteligencia artificial", "nutrición"],
        "texto": (
            "El calentamiento global es uno de los mayores desafíos del siglo XXI. "
            "Las temperaturas promedio han aumentado aproximadamente 1.1 grados Celsius desde la era preindustrial. "
            "El derretimiento de los glaciares eleva el nivel del mar y amenaza a millones de personas en zonas costeras. "
            "Las olas de calor son cada vez más frecuentes e intensas en todo el planeta. "
            "La inteligencia artificial ha transformado radicalmente la industria tecnológica en la última década. "
            "Los modelos de lenguaje grandes pueden generar texto, código y análisis con una precisión sorprendente. "
            "Empresas de todo el mundo compiten por desarrollar sistemas más capaces y eficientes. "
            "La automatización impulsada por IA está redefiniendo el mercado laboral. "
            "La dieta mediterránea es reconocida mundialmente por sus beneficios para la salud cardiovascular. "
            "Se basa en el consumo abundante de frutas, verduras, legumbres y aceite de oliva. "
            "El consumo moderado de pescado y vino tinto también forma parte de sus principios. "
            "Numerosos estudios clínicos confirman que reduce el riesgo de enfermedades crónicas."
        ),
        "segmentos": [
            [
                "El calentamiento global es uno de los mayores desafíos del siglo XXI.",
                "Las temperaturas promedio han aumentado aproximadamente 1.1 grados Celsius desde la era preindustrial.",
                "El derretimiento de los glaciares eleva el nivel del mar y amenaza a millones de personas en zonas costeras.",
                "Las olas de calor son cada vez más frecuentes e intensas en todo el planeta.",
            ],
            [
                "La inteligencia artificial ha transformado radicalmente la industria tecnológica en la última década.",
                "Los modelos de lenguaje grandes pueden generar texto, código y análisis con una precisión sorprendente.",
                "Empresas de todo el mundo compiten por desarrollar sistemas más capaces y eficientes.",
                "La automatización impulsada por IA está redefiniendo el mercado laboral.",
            ],
            [
                "La dieta mediterránea es reconocida mundialmente por sus beneficios para la salud cardiovascular.",
                "Se basa en el consumo abundante de frutas, verduras, legumbres y aceite de oliva.",
                "El consumo moderado de pescado y vino tinto también forma parte de sus principios.",
                "Numerosos estudios clínicos confirman que reduce el riesgo de enfermedades crónicas.",
            ],
        ],
    },
    {
        "id": "D02",
        "temas": ["astronomía", "economía", "deporte"],
        "texto": (
            "El universo tiene aproximadamente 13.800 millones de años desde el Big Bang. "
            "Las galaxias se agrupan en filamentos cósmicos separados por enormes vacíos. "
            "Los agujeros negros supermasivos residen en el centro de la mayoría de las galaxias. "
            "La expansión del universo se acelera debido a la energía oscura. "
            "La inflación es el aumento sostenido del nivel general de precios de bienes y servicios. "
            "Los bancos centrales utilizan las tasas de interés para controlar la inflación. "
            "Una inflación elevada reduce el poder adquisitivo de los hogares. "
            "Las políticas fiscales y monetarias deben coordinarse para mantener la estabilidad económica. "
            "El fútbol es el deporte más popular del mundo con más de cuatro mil millones de seguidores. "
            "La Copa del Mundo se celebra cada cuatro años y reúne a 32 selecciones nacionales. "
            "El entrenamiento físico de los futbolistas profesionales incluye resistencia, velocidad y táctica. "
            "Las lesiones musculares son las más comunes en este deporte de alto rendimiento."
        ),
        "segmentos": [
            [
                "El universo tiene aproximadamente 13.800 millones de años desde el Big Bang.",
                "Las galaxias se agrupan en filamentos cósmicos separados por enormes vacíos.",
                "Los agujeros negros supermasivos residen en el centro de la mayoría de las galaxias.",
                "La expansión del universo se acelera debido a la energía oscura.",
            ],
            [
                "La inflación es el aumento sostenido del nivel general de precios de bienes y servicios.",
                "Los bancos centrales utilizan las tasas de interés para controlar la inflación.",
                "Una inflación elevada reduce el poder adquisitivo de los hogares.",
                "Las políticas fiscales y monetarias deben coordinarse para mantener la estabilidad económica.",
            ],
            [
                "El fútbol es el deporte más popular del mundo con más de cuatro mil millones de seguidores.",
                "La Copa del Mundo se celebra cada cuatro años y reúne a 32 selecciones nacionales.",
                "El entrenamiento físico de los futbolistas profesionales incluye resistencia, velocidad y táctica.",
                "Las lesiones musculares son las más comunes en este deporte de alto rendimiento.",
            ],
        ],
    },
    {
        "id": "D03",
        "temas": ["historia", "biología celular"],
        "texto": (
            "La Segunda Guerra Mundial fue el conflicto bélico más devastador de la historia humana. "
            "Se extendió entre 1939 y 1945 e involucró a la mayoría de las naciones del mundo. "
            "El Holocausto resultó en el asesinato sistemático de seis millones de judíos europeos. "
            "La guerra concluyó con la rendición de Alemania en mayo y de Japón en septiembre de 1945. "
            "La célula es la unidad fundamental de la vida en todos los organismos conocidos. "
            "Las células eucariotas contienen un núcleo delimitado por una membrana nuclear. "
            "La mitosis permite que una célula madre genere dos células hijas genéticamente idénticas. "
            "Las mitocondrias producen energía en forma de ATP mediante la respiración celular."
        ),
        "segmentos": [
            [
                "La Segunda Guerra Mundial fue el conflicto bélico más devastador de la historia humana.",
                "Se extendió entre 1939 y 1945 e involucró a la mayoría de las naciones del mundo.",
                "El Holocausto resultó en el asesinato sistemático de seis millones de judíos europeos.",
                "La guerra concluyó con la rendición de Alemania en mayo y de Japón en septiembre de 1945.",
            ],
            [
                "La célula es la unidad fundamental de la vida en todos los organismos conocidos.",
                "Las células eucariotas contienen un núcleo delimitado por una membrana nuclear.",
                "La mitosis permite que una célula madre genere dos células hijas genéticamente idénticas.",
                "Las mitocondrias producen energía en forma de ATP mediante la respiración celular.",
            ],
        ],
    },
    {
        "id": "D04",
        "temas": ["música", "arquitectura", "psicología"],
        "texto": (
            "El jazz surgió a finales del siglo XIX en Nueva Orleans a partir de tradiciones africanas y europeas. "
            "Se caracteriza por la improvisación, el swing y el uso de escalas pentatónicas y de blues. "
            "Miles Davis y John Coltrane fueron figuras clave en la evolución del jazz moderno. "
            "El género influyó profundamente en el rock, el soul y la música electrónica contemporánea. "
            "La arquitectura gótica se desarrolló en Europa occidental entre los siglos XII y XVI. "
            "Sus elementos distintivos incluyen arcos ojivales, arbotantes y grandes vidrieras de colores. "
            "La catedral de Notre-Dame de París es uno de los ejemplos más representativos del estilo. "
            "El objetivo era crear espacios que condujeran la luz hacia el interior del templo. "
            "La psicología cognitiva estudia los procesos mentales como la memoria, el lenguaje y el pensamiento. "
            "Jean Piaget describió etapas del desarrollo cognitivo en la infancia y adolescencia. "
            "Los sesgos cognitivos son errores sistemáticos del razonamiento que afectan nuestras decisiones. "
            "La terapia cognitivo-conductual es una de las intervenciones psicológicas con mayor evidencia empírica."
        ),
        "segmentos": [
            [
                "El jazz surgió a finales del siglo XIX en Nueva Orleans a partir de tradiciones africanas y europeas.",
                "Se caracteriza por la improvisación, el swing y el uso de escalas pentatónicas y de blues.",
                "Miles Davis y John Coltrane fueron figuras clave en la evolución del jazz moderno.",
                "El género influyó profundamente en el rock, el soul y la música electrónica contemporánea.",
            ],
            [
                "La arquitectura gótica se desarrolló en Europa occidental entre los siglos XII y XVI.",
                "Sus elementos distintivos incluyen arcos ojivales, arbotantes y grandes vidrieras de colores.",
                "La catedral de Notre-Dame de París es uno de los ejemplos más representativos del estilo.",
                "El objetivo era crear espacios que condujeran la luz hacia el interior del templo.",
            ],
            [
                "La psicología cognitiva estudia los procesos mentales como la memoria, el lenguaje y el pensamiento.",
                "Jean Piaget describió etapas del desarrollo cognitivo en la infancia y adolescencia.",
                "Los sesgos cognitivos son errores sistemáticos del razonamiento que afectan nuestras decisiones.",
                "La terapia cognitivo-conductual es una de las intervenciones psicológicas con mayor evidencia empírica.",
            ],
        ],
    },
    {
        "id": "D05",
        "temas": ["energía renovable", "derecho", "gastronomía"],
        "texto": (
            "La energía solar fotovoltaica convierte la radiación solar directamente en electricidad. "
            "Los paneles solares han reducido su costo en más del 90% durante la última década. "
            "La energía eólica aprovecha el viento para mover turbinas conectadas a generadores eléctricos. "
            "Ambas fuentes son clave para reducir las emisiones de gases de efecto invernadero. "
            "El derecho internacional regula las relaciones entre Estados soberanos y organizaciones internacionales. "
            "La Corte Internacional de Justicia resuelve disputas entre países miembros de la ONU. "
            "Los tratados bilaterales y multilaterales constituyen la principal fuente del derecho internacional. "
            "El principio de soberanía estatal es el fundamento del orden jurídico internacional moderno. "
            "La gastronomía molecular aplica técnicas científicas para transformar texturas y sabores. "
            "Ferran Adrià popularizó el uso de la esferificación y las espumas en la cocina de vanguardia. "
            "El nitrógeno líquido permite congelar alimentos instantáneamente sin formar cristales de hielo. "
            "Estos métodos buscan crear experiencias sensoriales únicas más allá de la nutrición tradicional."
        ),
        "segmentos": [
            [
                "La energía solar fotovoltaica convierte la radiación solar directamente en electricidad.",
                "Los paneles solares han reducido su costo en más del 90% durante la última década.",
                "La energía eólica aprovecha el viento para mover turbinas conectadas a generadores eléctricos.",
                "Ambas fuentes son clave para reducir las emisiones de gases de efecto invernadero.",
            ],
            [
                "El derecho internacional regula las relaciones entre Estados soberanos y organizaciones internacionales.",
                "La Corte Internacional de Justicia resuelve disputas entre países miembros de la ONU.",
                "Los tratados bilaterales y multilaterales constituyen la principal fuente del derecho internacional.",
                "El principio de soberanía estatal es el fundamento del orden jurídico internacional moderno.",
            ],
            [
                "La gastronomía molecular aplica técnicas científicas para transformar texturas y sabores.",
                "Ferran Adrià popularizó el uso de la esferificación y las espumas en la cocina de vanguardia.",
                "El nitrógeno líquido permite congelar alimentos instantáneamente sin formar cristales de hielo.",
                "Estos métodos buscan crear experiencias sensoriales únicas más allá de la nutrición tradicional.",
            ],
        ],
    },
]


def get_all() -> list:
    return DATASET


def get_by_id(doc_id: str) -> dict:
    for doc in DATASET:
        if doc["id"] == doc_id:
            return doc
    raise KeyError(f"No existe documento con id={doc_id}")


def boundaries_from_segments(segments: list) -> list:
    """Devuelve índices de frontera (posición de la última oración de cada segmento excepto el último)."""
    boundaries = []
    idx = 0
    for seg in segments[:-1]:
        idx += len(seg)
        boundaries.append(idx - 1)
    return boundaries
