const currentDate = new Date();
const peopleCount = 11;

export const People = (new Array(peopleCount)).map((value, index) => {
  return {
    id: index,
    photo: `assets/photos/${index + 1}.jpg`,
    rate: Math.floor(Math.random() * 5),
    description: `Rate ${Math.floor(Math.random() * 5)}`
  };
});
