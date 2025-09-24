import { Box, HStack, Link, Text } from '@chakra-ui/react'
import { useColorPalette, useFont } from '../Helper/Utils'
import { ColorModeButton } from '../ui/color-mode'


const Dashboard_Nav = () => {
  const colors = useColorPalette()
  const fontfa = useFont()
  return (
    <Box p={"10px"} w={"99vw"} border={"1px solid"} rounded={"7px"} mx={"10px"}>
        <HStack w={'100%'} justifyContent={"space-between"} alignItems={"center"}>
          <Link href="/home">
            <Text fontFamily={fontfa.family_1} fontSize={"18px"}>Home</Text>
          </Link>
          <Link href='/home/languages'><Text fontSize={"20px"} fontFamily={fontfa.family_1}>Learning Languages</Text></Link>
          <Link href='/home/healthcare'><Text fontSize={"20px"} fontFamily={fontfa.family_1} >Examining Health</Text></Link>
          <Link href='/home/forex'><Text fontSize={"20px"} fontFamily={fontfa.family_1}>Foxex</Text></Link>
          <ColorModeButton/>
        </HStack>
    </Box>
  )
}

export default Dashboard_Nav 