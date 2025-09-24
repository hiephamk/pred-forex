import { Box, VStack, HStack, Heading, Text } from '@chakra-ui/react'
import { Outlet } from 'react-router'
import { useColorPalette, useFont } from '../components/Helper/Utils'

const LanguagePage = () => {
  const colors = useColorPalette()
  const fontfa = useFont()
  return (
    <Box>
        <HStack justifyContent={"space-between"} mx={"5px"}>
          <VStack flexBasis={"25%"} h={"100vh"} p={"10px"}>
            <Text fontSize={"20px"} fontFamily={fontfa.family_1}>My Leanring</Text>
          </VStack>
          <VStack p={"10px"} flexBasis={"75%"} rounded={"7px"} h={"100vh"} maxH={"100vh"} overflow={"auto"} border={"1px solid"}>
            <Outlet/>
          </VStack>
          <VStack flexBasis={"25%"} h={"100vh"} fontFamily={fontfa.family_1} fontSize={"20px"}>
            <Text>My Achievement</Text>
          </VStack>
        </HStack>
    </Box>
  )
}

export default LanguagePage